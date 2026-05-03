import numpy as np
import yaml
import os
from typing import List, Dict, Tuple, Any
import cv2
import json

from src.DataCleanModel.model.sam3_concept_prompt_model import Sam3ConceptPromptInference
from src.DataCleanModel.partition_merge.partition import naive_frame_partition, device_file_frame_partition
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo, LabelInfo, AnnotationFileGenerator, create_thickness_polygon
from src.DataCleanModel.utils.ai_labeler_json_processer import read_json_data,  \
    generate_instance_mask, find_the_most_frequent_shape, approximate_mask_with_shape,generate_roi
from src.DataCleanModel.utils.mask_matcher import mask_matcher
from src.DataCleanModel.utils.metrics import  calculate_hausdorff_distance
from src.DataCleanModel.utils.mask_merge import mask_merge,is_close_to_boundary
from src.DataCleanModel.partition_merge.partition import Partition
from src.DataCleanModel.partition_merge.ga_based_bin_pack_algorithm import partition_optimization_ga
from src.DataCleanModel.utils.metrics import compute_sam3_precison_recall_with_gt


  
class DataValidationWithFoundationModel:
    def __init__(self, config_source: str | dict[str, Any]):
        if isinstance(config_source, str):
            if not os.path.isfile(config_source):
                raise FileNotFoundError(f"Config file not found: {config_source}")
            with open(config_source) as f:
                data = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            data = config_source
        else:
            raise ValueError("Invalid config source type. Must be a file path or a dictionary.")

        self.model_cfg_path = data["model_cfg_path"]
        self.iou_score_threshold = data["iou_score_threshold"]              # IoU score threshold for reporting low quality annotation
        self.hausdorff_distance_threshold = data["hausdorff_distance_threshold"]    # Hausdorff distance threshold for reporting low quality annotation
        self.labels = data["labels"]                                 # List of labels to be processed, can be adjusted based on actual use case
        self.image_name_list = data["image_list"]
        self.frame_roi = data["frame_roi"]
        self.contour_approximation_epsilon = data["contour_approximation_epsilon"] # Epsilon value for contour approximation, can be adjusted based on actual use case, smaller value will result in more detailed polygon, larger value will result in simpler polygon.
        self.merge_mask_flag = data["merge_mask_flag"] # Merge boundary mask into the original mask for evaluation, can be set to False to keep them separate. If set to True, the boundary mask will be merged into the original mask with a new label value specified by "merge_boundary_label_value".
        self.merge_mask_min_overlap_area = data["merge_mask_min_overlap_area"]
        self.merge_mask_min_iou = data["merge_mask_min_iou"]
        self.naive_partition_overlap = data["naive_partition_overlap"]
        self.exemplar_roi_expansion = data["exemplar_roi_expansion"]
        self.iou_low_quality_threshold=data["iou_low_quality_threshold"]
        self.merge_mask_min_partial_overlapping_ratio = data["merge_mask_min_partial_overlapping_ratio"]
        self.prediction_score_threshold = data["prediction_score_threshold"]
        self.labels_max_position_shift = data["labels_max_position_shift"]
        self.boundary_mask_tolerance = data["boundary_mask_tolerance"]
        self.enable_merge_after_partition = data["enable_merge_after_partition"]
        self.partition_merge_gap = data["partition_merge_gap"]
        self.target_wire_width = data["target_wire_width"]
        # Hardcoded to use SAM3 concept prompt model for now, can be extended to support more foundation models in the future
        self.inference_model = Sam3ConceptPromptInference(self.model_cfg_path)
        self.model_input_size = (1008, 1008)
        
    def _merge_image_list(self, 
                          image_list:Dict[str, np.ndarray]) -> np.ndarray:
        # get the shape of the first image in the list, and create a blank image with the same shape to merge all images in the list, the merged image will be used for inference
        h,w,_ = image_list[self.image_name_list[0]].shape
        merged_image = np.zeros((h, w, 3), dtype=image_list[self.image_name_list[0]].dtype)
        counter = 0
        for img_nm in self.image_name_list:
            # convert color to gray level image
            if image_list[img_nm].shape[2] == 3:
                merged_image[:, :, counter] = cv2.cvtColor(image_list[img_nm], cv2.COLOR_BGR2GRAY)
            else:
                merged_image[:, :, counter] = np.squeeze(image_list[img_nm],axis=-1)
            counter += 1
        return merged_image
    
    def _validate_strategy(self, strategy:str):
        supported_strategies = ["first"]
        if strategy not in supported_strategies:
            raise ValueError(f"Unsupported strategy: {strategy}. Supported strategies are: {supported_strategies}")
        
    def _locate_least_overlapped_hv_roi(self, rois:list[Tuple[int,int,int,int]]) -> Tuple[int, int]:
        rois_by_id = {idx: roi for idx, roi in enumerate(rois)}
        # group rois into two groups, one group is for horizontal rois and the other group is for vertical rois, the horizontal rois are defined as the rois with width greater than height, and the vertical rois are defined as the rois with height greater than width
        horizontal_rois = []
        vertical_rois = []
        for idx, roi in rois_by_id.items():
            x_min, y_min, x_max, y_max = roi
            width = x_max - x_min
            height = y_max - y_min
            if width > height:
                horizontal_rois.append((idx, roi))
            else:
                vertical_rois.append((idx, roi))

        # find the least overlapped roi in horizontal rois and vertical rois respectively, the least overlapped roi is defined as the roi with the smallest average overlap area with other rois in the same group, and return the one with smaller average overlap area between the least overlapped horizontal roi and the least overlapped vertical roi
        def compute_iou(boxA, boxB):
            # Determine the coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # Compute the area of intersection
            # max(0, ...) handles cases where boxes do not overlap
            interWidth = max(0, xB - xA)
            interHeight = max(0, yB - yA)
            interArea = interWidth * interHeight

            # Compute the area of both bounding boxes
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            # Compute IoU (Area of Union = sum of areas - intersection area)
            iou = interArea / float(boxAArea + boxBArea - interArea)

            return iou
        
        if len(rois) == 0:
            return None, None 

        if len(horizontal_rois) > 0:
            iou_matrix = np.zeros((len(horizontal_rois), len(horizontal_rois)), dtype=np.float32)

            for idx, (roi_idx, roi) in enumerate(horizontal_rois):
                x_min, y_min, x_max, y_max = roi
                for other_idx, (other_roi_idx, other_roi) in enumerate(horizontal_rois):
                    if idx == other_idx:
                        continue
                    iou = compute_iou(roi, other_roi)
                    iou_matrix[idx, other_idx] = iou
            h_max_iou_per_roi = np.max(iou_matrix, axis=1)
            least_overlapped_horizontal_idx = horizontal_rois[np.argmin(h_max_iou_per_roi)][0]
        else:
            least_overlapped_horizontal_idx = None

        if len(vertical_rois) > 0:
            iou_matrix = np.zeros((len(vertical_rois), len(vertical_rois)), dtype=np.float32)
            for idx, (roi_idx, roi) in enumerate(vertical_rois):
                x_min, y_min, x_max, y_max = roi
                for other_idx, (other_roi_idx, other_roi) in enumerate(vertical_rois):
                    if idx == other_idx:
                        continue
                    iou = compute_iou(roi, other_roi)
                    iou_matrix[idx, other_idx] = iou
            v_max_iou_per_roi = np.max(iou_matrix, axis=1)
            least_overlapped_vertical_idx = vertical_rois[np.argmin(v_max_iou_per_roi)][0]
        else:
            least_overlapped_vertical_idx = None

        return least_overlapped_horizontal_idx , least_overlapped_vertical_idx
       
    def compute_scale_factor(self, json_data):
        wire_width = float(json_data["globalAttributes"]["wire width"])
        if wire_width <= 0:
            return 1.0
        else:
            scale = self.target_wire_width / wire_width
            # limit scale factor to be between 0.5 and 2.0 to avoid extreme scaling that may cause issues in partition and inference
            scale = max(min(scale, 2.0), 0.5)
            return scale
        
    def prepare_for_frame_with_bond_program(self, 
                                            image: np.ndarray, 
                                            json_data: Dict,
                                            model_input_size_tolerance:float=5, 
                                            log_folder:str=None) -> Tuple[List[Partition], np.ndarray, List[Tuple[int,int,int,int]]]:
        object_rois = []
        height, width, _ = image.shape
        scale_factor = self.compute_scale_factor(json_data)

        resize_img = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)

        shift = int(self.labels_max_position_shift[0]) if len(self.labels_max_position_shift) > 0 else 0

        resize_bond_program = []
        for bond in json_data["globalAttributes"]["bond program"]:
            if len(bond) != 2:
                continue
            point1 = bond[0]
            point2 = bond[1]
            # extend the point1 and point2 by 25 pixel along the direction from point1 to point2
            direction = (point2[0] - point1[0], point2[1] - point1[1])
            length = (direction[0]**2 + direction[1]**2)**0.5
            if length > 0:
                extension = 60 
                direction = (direction[0] / length * extension, direction[1] / length * extension)
                point1 = (int(max(0, point1[0] - direction[0])), int(max(0, point1[1] - direction[1])))
                point2 = (int(min(width, point2[0] + direction[0])), int(min(height, point2[1] + direction[1])))
            x0 = min(point1[0], point2[0])
            y0 = min(point1[1], point2[1])
            x1 = max(point1[0], point2[0])
            y1 = max(point1[1], point2[1])
            x0 = int(max(x0 - model_input_size_tolerance, 0) * scale_factor)
            y0 = int(max(y0 - model_input_size_tolerance, 0) * scale_factor)
            x1 = int(min(x1 + model_input_size_tolerance, width) * scale_factor)
            y1 = int(min(y1 + model_input_size_tolerance, height) * scale_factor)
            # ignore those roi larger than model input size
            if x1 - x0 > self.model_input_size[0] - 2 * model_input_size_tolerance or y1 - y0 > self.model_input_size[1] - 2 * model_input_size_tolerance:
                continue
            object_rois.append((max(0, x0 - shift), max(0, y0 - shift), min(int(width * scale_factor), x1 + shift), min(int(height * scale_factor), y1 + shift)))
            resize_bond_program.append([[point1[0]*scale_factor, point1[1]*scale_factor], [point2[0]*scale_factor, point2[1]*scale_factor]])
        h_exemplar_idx, v_exemplar_idx = self._locate_least_overlapped_hv_roi(object_rois)
        if h_exemplar_idx is None and v_exemplar_idx is None:
            print("No valid exemplar roi found for current frame, cannot prepare for partition with bond program.")
            return None, None, None

        if log_folder is not None:
            # overlay exemplar roi on merged image and save for debugging
            debug_image = image.copy()
            debug_image =cv2.resize(debug_image, (int(debug_image.shape[1] * scale_factor), int(debug_image.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
            if h_exemplar_idx is not None:
                cv2.rectangle(debug_image, (object_rois[h_exemplar_idx][0], object_rois[h_exemplar_idx][1]), (object_rois[h_exemplar_idx][2], object_rois[h_exemplar_idx][3]), (0, 255, 0), 2)
            if v_exemplar_idx is not None:
                cv2.rectangle(debug_image, (object_rois[v_exemplar_idx][0], object_rois[v_exemplar_idx][1]), (object_rois[v_exemplar_idx][2], object_rois[v_exemplar_idx][3]), (255, 0, 0), 2)
            for bond in resize_bond_program:
                cv2.line(debug_image, (int(bond[0][0]), int(bond[0][1])), (int(bond[1][0]), int(bond[1][1])), (0, 255, 255), 2)
            cv2.imwrite(os.path.join(log_folder, f"exemplar_bond_program_debug_image.jpg"), debug_image[:,:,::-1])
        exmplar_rois = []
        if h_exemplar_idx is not None:
            exmplar_rois.append(object_rois[h_exemplar_idx])
        elif v_exemplar_idx is not None:
            exmplar_rois.append(object_rois[v_exemplar_idx])

        return device_file_frame_partition(frame_size=(image.shape[1], image.shape[0]),
                                            object_rois=object_rois,
                                            target_size=(self.model_input_size[0] - 2 * model_input_size_tolerance, 
                                                         self.model_input_size[1] - 2 * model_input_size_tolerance)), \
                                                                  exmplar_rois, resize_img

    def select_least_overlapped_object_roi_with_json(self, json_data:Dict, label_name_list:List[str], roi_expansion:int=5)->Tuple[int,int,int,int]:
        object_rois = []
        json_array,_,_ = read_json_data(json_data, label_name_list)
        for label_name in label_name_list:
            object_rois.extend(generate_roi(json_array, label_name, shift=roi_expansion))
        h_exemplar_idx, v_exemplar_idx = self._locate_least_overlapped_hv_roi(object_rois)
        if h_exemplar_idx is None and v_exemplar_idx is None:
            print("No valid exemplar roi found for current frame, cannot select exemplar roi with json.")
            return None
        if h_exemplar_idx is not None:
            return object_rois[h_exemplar_idx]
        elif v_exemplar_idx is not None:
            return object_rois[v_exemplar_idx]

    def data_curation_with_concept_prompt_bond_program(self,
                    inspection_path: str,   # path to inspection data 
                    image,  # color image
                    json_data:Dict,
                    partitions:list[Partition],
                    support_image:np.ndarray,
                    exemplar_rois:List[Tuple[float,float,float,float]],
                    log_folder:str=None):   
        # create log_folder
        if log_folder is not None and not os.path.exists(log_folder):
            os.makedirs(log_folder)

        scale = self.compute_scale_factor(json_data)

        resize_image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
        # load json_data to instance array
        json_array, json_array_remain, unique_label_name = read_json_data(json_data, self.labels, scale=scale)
        if json_array is None or len(json_array) == 0:
            print("No valid annotation found in json data for cross validation.")
            return
        
        # find most frequent shape of every label
        # approximate the shape by the most frequent shape
        most_frequent_shape = {}
        for label_name in self.labels:
            most_frequent_shape[label_name] = find_the_most_frequent_shape(json_array, label_name)

        # merge image if multiple images provided for one case
        merged_image = resize_image 

        ga_bins = None
        if partitions is not None and self.enable_merge_after_partition:
            ga_bins, ga_fit, _chr, best_fitness_history, mean_fitness_history = partition_optimization_ga(
                partitions,
                bin_width=self.model_input_size[0],
                bin_height=self.model_input_size[1],
                population_size=50,
                generations=20,
                selection_strategy="tournament",
                tournament_size=3,
                sus_selection_num=5,
                crossover_rate=0.9,
                swap_mutation_rate=0.20,
                inversion_mutation_rate=0.10,
                rotation_flip_rate=0.15,
                elitism=2,
                utilization_power_k=2.0,
                log_folder=log_folder,
                merge_gap=self.partition_merge_gap,
                image=merged_image)
            print(f"GA Optimization - Bins used: {len(ga_bins)}, Fitness: {ga_fit:.6f}")

        target_images = []
        target_images_offset = []
        if ga_bins is None:
            for partition in partitions:
                center = partition.center
                crop_ul = (max(int(center[0] - self.model_input_size[0]// 2), 0), max(int(center[1] - self.model_input_size[1] // 2), 0))
                crop_lr = (min(int(center[0] + self.model_input_size[0] // 2), merged_image.shape[1]), min(int(center[1] + self.model_input_size[1] // 2), merged_image.shape[0]))
                crop_image = merged_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0],:]
                # pad crop image to model input size if necessary
                if crop_image.shape[0] < self.model_input_size[1] or crop_image.shape[1] < self.model_input_size[0]:
                    padded_image = np.zeros((self.model_input_size[1], self.model_input_size[0], 3), dtype=crop_image.dtype)
                    padded_image[0:crop_image.shape[0], 0:crop_image.shape[1],:] = crop_image
                    crop_image = padded_image

                partition.offset.append(crop_ul)

                partition.opsize.append((crop_lr[0] - crop_ul[0], crop_lr[1] - crop_ul[1]))
                target_images.append(crop_image)
                target_images_offset.append(crop_ul)
                #cv2.imwrite(os.path.join(log_folder, f"partition_{center[0]}_{center[1]}.jpg"), crop_image[:,:,::-1])
                print("Partition center: {}, crop_ul: {}, crop_lr: {}, size: {},{}".format(center, crop_ul, crop_lr, target_images[-1].shape[1], target_images[-1].shape[0]))
            if log_folder is not None:
                # overlay partition on merged image and save for debugging
                debug_image = merged_image.copy()
                for partition in partitions:
                    center = partition.center
                    cv2.circle(debug_image, (int(center[0]), int(center[1])), 5, (255, 255, 0), -1)
                    crop_ul = (max(int(center[0] - self.model_input_size[0]// 2), 0), max(int(center[1] - self.model_input_size[1] // 2), 0))
                    crop_lr = (min(int(center[0] + self.model_input_size[0] // 2), merged_image.shape[1]), min(int(center[1] + self.model_input_size[1] // 2), merged_image.shape[0]))
                    cv2.rectangle(debug_image, (crop_ul[0], crop_ul[1]), (crop_lr[0], crop_lr[1]), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(log_folder, "partition_debug_image.jpg"), debug_image[:,:,::-1])
        else:
            partition_by_id = {b.id: b for b in partitions}
            for bin in ga_bins:
                target_images.append(bin.form_bin_frame(merged_image,partition_by_id, self.partition_merge_gap))
        
        # perform inference for each label and compare with annotation to find low quality annotation
        results = {}
        for label_name in self.labels:
            gt_shape = []
            for item in json_array:
                if item.m_label == label_name:
                    gt_shape.append(item)
            print(f"Processing label: {label_name}, GT shape count: {len(gt_shape)}")
            pred_shape = []

            reference_image = np.zeros((self.model_input_size[0], self.model_input_size[1], 3), dtype=np.uint8)
            for roi_idx, exemplar_roi in enumerate(exemplar_rois):  
                # crop image with exemplar roi and perform inference, get predicted mask and predicted roi
                crop_center_x = (exemplar_roi[0] + exemplar_roi[2]) // 2
                crop_center_y = (exemplar_roi[1] + exemplar_roi[3]) // 2
                crop_ul = (max(crop_center_x - self.model_input_size[0]//2, 0), max(crop_center_y - self.model_input_size[1]//2, 0))
                crop_lr = (min(crop_center_x + self.model_input_size[0]//2, merged_image.shape[1]), min(crop_center_y + self.model_input_size[1]//2, merged_image.shape[0]))
                cropped_image = support_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0],:]
                # paste cropp image to reference image center
                reference_image_center = (self.model_input_size[0] // 2, self.model_input_size[1] // 2)
                cropped_image_center = (cropped_image.shape[0] // 2, cropped_image.shape[1] // 2)
                start_y = max(reference_image_center[0] - cropped_image_center[0], 0)
                start_x = max(reference_image_center[1] - cropped_image_center[1], 0)
                end_y = start_y + cropped_image.shape[0]
                end_x = start_x + cropped_image.shape[1]
                reference_image[start_y:end_y, start_x:end_x, :] = cropped_image
                exemplar_roi_in_reference = {"x": exemplar_roi[0] - crop_ul[0], "y": exemplar_roi[1] - crop_ul[1], "width": exemplar_roi[2] - exemplar_roi[0], "height": exemplar_roi[3] - exemplar_roi[1]}
                # prepare reference state
                ref_state = self.inference_model.prepare_reference_state(reference_image, 
                                                                        [exemplar_roi_in_reference], 
                                                                        [True],
                                                                        label_name.lower(), 
                                                                        True)
                if ref_state is None:
                    continue

                if ga_bins is None:
                    for offset, target_image, partition in zip(target_images_offset, target_images, partitions):
                        # display progress of inference
                        annotated_target, _ = self.inference_model.run_cross_inference(target_image,
                                                                                        ref_state,
                                                                                        self.prediction_score_threshold,
                                                                                        self.prediction_score_threshold)
                        annotation = annotated_target[1]
                        for mask, _, score in annotation: 
                            s_type, approx_points = approximate_mask_with_shape(mask, offset, most_frequent_shape[label_name],self.contour_approximation_epsilon)
                            if s_type is not None and len(approx_points) >= 4:
                                s = ShapeInfo(type=s_type)
                                s.set_general_shape_point(approx_points, label_name)    
                                s.m_score = score 
                                # remove boundary mask      
                                if not is_close_to_boundary(s, partition.offset[0], partition.opsize[0], 1):       
                                    pred_shape.append(s)

                else:
                    for i, (target_image, bin) in enumerate(zip(target_images, ga_bins)):
                        bin_pred_shape = []
                        # display progress of inference
                        annotated_target, _ = self.inference_model.run_cross_inference(target_image,
                                                                                        ref_state,
                                                                                        self.prediction_score_threshold,
                                                                                        self.prediction_score_threshold)
                        annotation = annotated_target[1]
                        for mask, _, score in annotation: 
                            s_type, approx_points = approximate_mask_with_shape(mask, (0,0), most_frequent_shape[label_name],self.contour_approximation_epsilon)
                            # check whether the approx points touch the partition's boundary in bin, if yes, consider it as boundary mask and remove it
                            if s_type is not None and len(approx_points) >= 4:
                                s = ShapeInfo(type=s_type)
                                s.set_general_shape_point(approx_points, label_name)    
                                s.m_score = score
                                if not bin.is_close_to_boundary(s, self.partition_merge_gap):
                                    bin_pred_shape.append(s)
                
                        if log_folder:
                            # save predicted mask for debugging
                            debug_image = target_image.copy()
                            for s in bin_pred_shape: 
                                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                                debug_image = s.draw_shape(debug_image, color)
                                cv2.putText(debug_image, f"{s.m_score:.2f}", (int(s.m_points[0][0]), int(s.m_points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            cv2.imwrite(os.path.join(log_folder, f"predicted_mask_bin_{i}_{label_name}_roi_{roi_idx}.jpg"), debug_image[:,:,::-1])
                    
                        pred_shape.extend(bin.decompose_prediction_results(bin_pred_shape, partition_by_id, self.partition_merge_gap)) 

                if log_folder:
                    # save predicted mask for debugging
                    debug_image = merged_image.copy()
                    for s in pred_shape: 
                        # assign random color to pred shape for better visualization
                        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        debug_image = s.draw_shape(debug_image, color)
                        cv2.putText(debug_image, f"{s.m_score:.2f}", (int(s.m_points[0][0]), int(s.m_points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    #for s in gt_shape:
                    #    debug_image = s.draw_shape(debug_image, (255, 255, 0)) 
                    cv2.imwrite(os.path.join(log_folder, f"predicted_mask_{label_name}.jpg"), debug_image[:,:,::-1])

            # mask redundancy filtering
            filtered_pred_shape = pred_shape
            for k in range(3):
                filtered_pred_shape = mask_merge(filtered_pred_shape,
                            nms_iou_threshold=self.merge_mask_min_iou,
                            min_intersection_area=self.merge_mask_min_overlap_area,
                            min_partial_overlapping_ratio=self.merge_mask_min_partial_overlapping_ratio,
                            boundary_mask_removed=True)
            if log_folder:
                # save predicted mask for debugging
                debug_image = merged_image.copy()
                for s in filtered_pred_shape: 
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    debug_image = s.draw_shape(debug_image, color)
                    cv2.putText(debug_image, f"{s.m_score:.2f}", (int(s.m_points[0][0]), int(s.m_points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                #for s in gt_shape:
                #    debug_image = s.draw_shape(debug_image, (255, 255, 0)) 
                cv2.imwrite(os.path.join(log_folder, f"filtered_predicted_mask_{label_name}.jpg"), debug_image[:,:,::-1])

            # match predicted masks with annotation masks
            matched_indices, unmatched_gt_indices, unmatched_pred_indices = mask_matcher(gt_shape, filtered_pred_shape, iou_threshold=self.iou_score_threshold)   
            results[label_name] = (gt_shape, filtered_pred_shape, matched_indices, unmatched_gt_indices, unmatched_pred_indices)
            
        # create updated label.json
        annotation_file = AnnotationFileGenerator()
        if log_folder:
            debug_image = image.copy()
        category_id = 0
        for label_name, (gt_shape, pred_shape, matched_indices, unmatched_gt_indices, unmatched_pred_indices) in results.items():
            label_info = LabelInfo(label_name)
            for gt_idx, pred_idx, iou in matched_indices:
                # scale back to original image coordinate
                points = pred_shape[pred_idx].m_points
                points = [[p[0] / scale, p[1] / scale] for p in points]
                if iou > 0.45:
                    if log_folder:
                        s = ShapeInfo(type=pred_shape[pred_idx].m_type)
                        s.set_general_shape_point(points, label_name)
                        debug_image = s.draw_shape(debug_image, (0, 255, 0))
                        cv2.putText(debug_image, f"{iou:.2f}", (int(points[0][0]), int(points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    gt_shape[gt_idx].m_points = points  # update annotation with predicted shape
                    gt_shape[gt_idx].m_attributes['category_id'] = category_id
                    category_id += 1
                    label_info.add_shape(gt_shape[gt_idx])
                else:
                    label_info.add_shape(gt_shape[gt_idx])  

            for pred_idx in unmatched_pred_indices:
                points = pred_shape[pred_idx].m_points
                points = [[p[0] / scale, p[1] / scale] for p in points]
                if log_folder:
                    s = ShapeInfo(type=pred_shape[pred_idx].m_type)
                    s.set_general_shape_point(points, label_name)
                    debug_image = s.draw_shape(debug_image, (0, 255, 0))
                    cv2.putText(debug_image, "unmatched", (int(points[0][0]), int(points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                pred_shape[pred_idx].m_points = points # trigger ai labler to display  
                pred_shape[pred_idx].m_attributes['category_id'] = category_id
                category_id += 1
                label_info.add_shape(pred_shape[pred_idx])  
            annotation_file.add_label_info(label_info)
            annotation_file.m_global_attributes = json_data.get("globalAttributes", {})
            annotation_file.add_global_attribute("exemplar_roi", exemplar_rois)
            
        annotation_file.serialize_to_json(inspection_path, "label_clean.json")
        if log_folder:
            cv2.imwrite(os.path.join(log_folder, f"final_updated_annotation.jpg"), debug_image[:,:,::-1])

    def get_image_embedding(self, img:np.ndarray):
        return self.inference_model.get_image_embedding(img)

    def data_validation_with_concept_prompt_bond_program(self,
                    inspection_path: str,   # path to inspection data 
                    image,  # color image
                    json_data:Dict,
                    partitions:list[Partition],
                    support_image:np.ndarray, 
                    exemplar_rois:List[Tuple[float,float,float,float]], 
                    result_folder:str,
                    log_folder:str=None):
        
        result_info = {"inspection_path": inspection_path}

        # create log_folder
        if log_folder is not None and not os.path.exists(log_folder):
            os.makedirs(log_folder)
        if result_folder is not None and not os.path.exists(result_folder):
            os.makedirs(result_folder)

        scale = self.compute_scale_factor(json_data)

        result_info["scale_factor"] = scale
        result_info["original_image_size"] = (image.shape[1], image.shape[0])

        resize_image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
        # load json_data to instance array
        json_array, _, _ = read_json_data(json_data, self.labels, scale=scale)
        if json_array is None or len(json_array) == 0:
            print("No valid annotation found in json data for cross validation.")
            return
        resized_exemplar_rois = []
        for exemplar_roi in exemplar_rois:
            resized_exemplar_roi = [int(coord * scale) for coord in exemplar_roi]
            resized_exemplar_rois.append(resized_exemplar_roi)
        resized_exemplar_img = cv2.resize(support_image, (int(support_image.shape[1]*scale), int(support_image.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)

        # visualize exemplar roi on support image and save for debugging
        if log_folder:
            debug_image = resized_exemplar_img.copy()
            for roi in resized_exemplar_rois:
                x1, y1, x2, y2 = roi
                cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(log_folder, f"exemplar_roi.jpg"), debug_image[:,:,::-1])
        
        # find most frequent shape of every label
        # approximate the shape by the most frequent shape
        most_frequent_shape = {}
        for label_name in self.labels:
            most_frequent_shape[label_name] = find_the_most_frequent_shape(json_array, label_name)

        # merge image if multiple images provided for one case
        merged_image = resize_image 

        ga_bins = None
        if partitions is not None and self.enable_merge_after_partition:
            ga_bins, ga_fit, _chr, best_fitness_history, mean_fitness_history = partition_optimization_ga(
                partitions,
                bin_width=self.model_input_size[0],
                bin_height=self.model_input_size[1],
                population_size=50,
                generations=20,
                selection_strategy="tournament",
                tournament_size=3,
                sus_selection_num=5,
                crossover_rate=0.9,
                swap_mutation_rate=0.20,
                inversion_mutation_rate=0.10,
                rotation_flip_rate=0.15,
                elitism=2,
                utilization_power_k=2.0,
                log_folder=log_folder,
                merge_gap=self.partition_merge_gap,
                image=merged_image)
            print(f"GA Optimization - Bins used: {len(ga_bins)}, Fitness: {ga_fit:.6f}")
        
        # compute partition number with grid partition with 10% overlap
        grid_partition_cols = int(np.ceil(merged_image.shape[1] / (self.model_input_size[0] * 0.9)))
        grid_partition_rows = int(np.ceil(merged_image.shape[0] / (self.model_input_size[1] * 0.9)))
        grid_partition_num = grid_partition_cols * grid_partition_rows
        result_info["grid_partition_num"] = grid_partition_num
        result_info["ga_bins_used"] = len(ga_bins) if ga_bins is not None else 0
        result_info["ga_fitness"] = ga_fit if ga_bins is not None else None
        result_info["partition_efficiency_score"] = grid_partition_num / float(len(ga_bins)) if ga_bins is not None else None

        target_images = []
        target_images_offset = []
        if ga_bins is None:
            for partition in partitions:
                center = partition.center
                crop_ul = (max(int(center[0] - self.model_input_size[0]// 2), 0), max(int(center[1] - self.model_input_size[1] // 2), 0))
                crop_lr = (min(int(center[0] + self.model_input_size[0] // 2), merged_image.shape[1]), min(int(center[1] + self.model_input_size[1] // 2), merged_image.shape[0]))
                crop_image = merged_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0],:]
                # pad crop image to model input size if necessary
                if crop_image.shape[0] < self.model_input_size[1] or crop_image.shape[1] < self.model_input_size[0]:
                    padded_image = np.zeros((self.model_input_size[1], self.model_input_size[0], 3), dtype=crop_image.dtype)
                    padded_image[0:crop_image.shape[0], 0:crop_image.shape[1],:] = crop_image
                    crop_image = padded_image

                partition.offset.append(crop_ul)

                partition.opsize.append((crop_lr[0] - crop_ul[0], crop_lr[1] - crop_ul[1]))
                target_images.append(crop_image)
                target_images_offset.append(crop_ul)
                #cv2.imwrite(os.path.join(log_folder, f"partition_{center[0]}_{center[1]}.jpg"), crop_image[:,:,::-1])
                print("Partition center: {}, crop_ul: {}, crop_lr: {}, size: {},{}".format(center, crop_ul, crop_lr, target_images[-1].shape[1], target_images[-1].shape[0]))
            if log_folder is not None:
                # overlay partition on merged image and save for debugging
                debug_image = merged_image.copy()
                for partition in partitions:
                    center = partition.center
                    cv2.circle(debug_image, (int(center[0]), int(center[1])), 5, (255, 255, 0), -1)
                    crop_ul = (max(int(center[0] - self.model_input_size[0]// 2), 0), max(int(center[1] - self.model_input_size[1] // 2), 0))
                    crop_lr = (min(int(center[0] + self.model_input_size[0] // 2), merged_image.shape[1]), min(int(center[1] + self.model_input_size[1] // 2), merged_image.shape[0]))
                    cv2.rectangle(debug_image, (crop_ul[0], crop_ul[1]), (crop_lr[0], crop_lr[1]), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(log_folder, "partition_debug_image.jpg"), debug_image[:,:,::-1])
        else:
            partition_by_id = {b.id: b for b in partitions}
            for bin in ga_bins:
                target_images.append(bin.form_bin_frame(merged_image,partition_by_id, self.partition_merge_gap))
        
        # perform inference for each label and compare with annotation to find low quality annotation
        results = {}
        for label_name in self.labels:
            gt_shape = []
            for item in json_array:
                if item.m_label == label_name:
                    gt_shape.append(item)
            print(f"Processing label: {label_name}, GT shape count: {len(gt_shape)}")
            pred_shape = []

            reference_image = np.zeros((self.model_input_size[0], self.model_input_size[1], 3), dtype=np.uint8)
            for roi_idx, exemplar_roi in enumerate(resized_exemplar_rois):  
                # crop image with exemplar roi and perform inference, get predicted mask and predicted roi
                crop_center_x = (exemplar_roi[0] + exemplar_roi[2]) // 2
                crop_center_y = (exemplar_roi[1] + exemplar_roi[3]) // 2
                crop_ul = (max(crop_center_x - self.model_input_size[0]//2, 0), max(crop_center_y - self.model_input_size[1]//2, 0))
                crop_lr = (min(crop_center_x + self.model_input_size[0]//2, merged_image.shape[1]), min(crop_center_y + self.model_input_size[1]//2, merged_image.shape[0]))
                cropped_image = resized_exemplar_img[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0],:]
                # paste cropp image to reference image center
                reference_image_center = (self.model_input_size[0] // 2, self.model_input_size[1] // 2)
                cropped_image_center = (cropped_image.shape[0] // 2, cropped_image.shape[1] // 2)
                start_y = max(reference_image_center[0] - cropped_image_center[0], 0)
                start_x = max(reference_image_center[1] - cropped_image_center[1], 0)
                end_y = start_y + cropped_image.shape[0]
                end_x = start_x + cropped_image.shape[1]
                reference_image[start_y:end_y, start_x:end_x, :] = cropped_image
                exemplar_roi_in_reference = {"x": exemplar_roi[0] - crop_ul[0] + start_x, "y": exemplar_roi[1] - crop_ul[1] + start_y, \
                                             "width": exemplar_roi[2] - exemplar_roi[0], "height": exemplar_roi[3] - exemplar_roi[1]}
                # save result exemplar info
                debug_image = reference_image.copy()
                cv2.rectangle(debug_image, (int(exemplar_roi_in_reference["x"]), int(exemplar_roi_in_reference["y"])), 
                                  (int(exemplar_roi_in_reference["x"] + exemplar_roi_in_reference["width"]), int(exemplar_roi_in_reference["y"] + exemplar_roi_in_reference["height"])), 
                                  (0, 255, 0), 2)
                cv2.imwrite(os.path.join(result_folder, f"reference_image_label_{label_name}_roi_{roi_idx}.jpg"), debug_image[:,:,::-1])
              
                if "exemplars_image" in result_info:
                    result_info["exemplars_image"].append(os.path.join(result_folder, f"reference_image_label_{label_name}_roi_{roi_idx}.jpg"))

                else:
                    result_info["exemplars_image"] = os.path.join(result_folder, f"reference_image_label_{label_name}_roi_{roi_idx}.jpg")

                # prepare reference state
                ref_state = self.inference_model.prepare_reference_state(reference_image, 
                                                                        [exemplar_roi_in_reference], 
                                                                        [True],
                                                                        label_name.lower(),
                                                                        True)
                if ref_state is None:
                    print(f"Reference state preparation failed for label {label_name} at roi index {roi_idx}. Skipping inference for this exemplar.")
                    continue

                if ga_bins is None:
                    for offset, target_image, partition in zip(target_images_offset, target_images, partitions):
                        # display progress of inference
                        annotated_target, _ = self.inference_model.run_cross_inference(target_image,
                                                                                        ref_state,
                                                                                        self.prediction_score_threshold,
                                                                                        self.prediction_score_threshold)
                        annotation = annotated_target[1]
                        for mask, _, score in annotation: 
                            s_type, approx_points = approximate_mask_with_shape(mask, offset, most_frequent_shape[label_name],self.contour_approximation_epsilon)
                            if s_type is not None and len(approx_points) >= 4:
                                s = ShapeInfo(type=s_type)
                                s.set_general_shape_point(approx_points, label_name)    
                                s.m_score = score 
                                # remove boundary mask      
                                if not is_close_to_boundary(s, partition.offset[0], partition.opsize[0], 1):       
                                    pred_shape.append(s)

                else:
                    for i, (target_image, bin) in enumerate(zip(target_images, ga_bins)):
                        bin_pred_shape = []
                        # display progress of inference
                        annotated_target, _ = self.inference_model.run_cross_inference(target_image,
                                                                                        ref_state,
                                                                                        self.prediction_score_threshold,
                                                                                        self.prediction_score_threshold)
                        annotation = annotated_target[1]
                        for mask, _, score in annotation: 
                            s_type, approx_points = approximate_mask_with_shape(mask, (0,0), most_frequent_shape[label_name],self.contour_approximation_epsilon)
                            # check whether the approx points touch the partition's boundary in bin, if yes, consider it as boundary mask and remove it
                            if s_type is not None and len(approx_points) >= 4:
                                s = ShapeInfo(type=s_type)
                                s.set_general_shape_point(approx_points, label_name)    
                                s.m_score = score
                                if not bin.is_close_to_boundary(s, self.partition_merge_gap):
                                    bin_pred_shape.append(s)
                
                        if log_folder:
                            # save predicted mask for debugging
                            debug_image = target_image.copy()
                            for s in bin_pred_shape: 
                                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                                debug_image = s.draw_shape(debug_image, color)
                                cv2.putText(debug_image, f"{s.m_score:.2f}", (int(s.m_points[0][0]), int(s.m_points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            cv2.imwrite(os.path.join(log_folder, f"predicted_mask_bin_{i}_{label_name}_roi_{roi_idx}.jpg"), debug_image[:,:,::-1])
                    
                        pred_shape.extend(bin.decompose_prediction_results(bin_pred_shape, partition_by_id, self.partition_merge_gap)) 

                if log_folder:
                    # save predicted mask for debugging
                    debug_image = merged_image.copy()
                    for s in pred_shape: 
                        # assign random color to pred shape for better visualization
                        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        debug_image = s.draw_shape(debug_image, color)
                        cv2.putText(debug_image, f"{s.m_score:.2f}", (int(s.m_points[0][0]), int(s.m_points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    #for s in gt_shape:
                    #    debug_image = s.draw_shape(debug_image, (255, 255, 0)) 
                    cv2.imwrite(os.path.join(log_folder, f"predicted_mask_{label_name}.jpg"), debug_image[:,:,::-1])

            # mask redundancy filtering
            filtered_pred_shape = pred_shape
            for k in range(3):
                filtered_pred_shape = mask_merge(filtered_pred_shape,
                            nms_iou_threshold=self.merge_mask_min_iou,
                            min_intersection_area=self.merge_mask_min_overlap_area,
                            min_partial_overlapping_ratio=self.merge_mask_min_partial_overlapping_ratio,
                            boundary_mask_removed=True)
            if log_folder:
                # save predicted mask for debugging
                debug_image = merged_image.copy()
                for s in filtered_pred_shape: 
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    debug_image = s.draw_shape(debug_image, color)
                    cv2.putText(debug_image, f"{s.m_score:.2f}", (int(s.m_points[0][0]), int(s.m_points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                #for s in gt_shape:
                #    debug_image = s.draw_shape(debug_image, (255, 255, 0)) 
                cv2.imwrite(os.path.join(log_folder, f"filtered_predicted_mask_{label_name}.jpg"), debug_image[:,:,::-1])

            # match predicted masks with annotation masks
            if log_folder:
                # overlay both gt_shape and filtered_pred_shape on original image and save for debugging
                debug_image = merged_image.copy()
                for s in gt_shape:
                    debug_image = s.draw_shape(debug_image, (0, 255, 0))
                for s in filtered_pred_shape:
                    debug_image = s.draw_shape(debug_image, (0, 0, 255))
                cv2.imwrite(os.path.join(log_folder, f"gt_vs_pred_{label_name}.jpg"), debug_image[:,:,::-1])

            matched_indices, unmatched_gt_indices, unmatched_pred_indices = mask_matcher(gt_shape, filtered_pred_shape, iou_threshold=self.iou_score_threshold)   
            results[label_name] = (gt_shape, filtered_pred_shape, matched_indices, unmatched_gt_indices, unmatched_pred_indices)

            result_info["Labels with Issue"] = []
            for label_name, (gt_shape, pred_shape, matched_indices, unmatched_gt_indices, unmatched_pred_indices) in results.items():
                for gt_idx, pred_idx, iou in matched_indices:
                    label_with_issue = {}
                    hausdorff_distance = calculate_hausdorff_distance(gt_shape[gt_idx], pred_shape[pred_idx])
                    if iou > self.iou_score_threshold and iou < self.iou_low_quality_threshold and hausdorff_distance > self.hausdorff_distance_threshold:  # threshold can be adjusted based on actual use case
                        #print(f"Low quality annotation found for label {label_name} at index {gt_idx} with IoU {iou} and Hausdorff distance {hausdorff_distance}.")
                        label_with_issue["id"] = gt_shape[gt_idx].m_attributes.get("category_id", "NA")
                        # overlay point and proposed point on original image and save for results review
                        debug_image = merged_image.copy()
                        debug_image = gt_shape[gt_idx].draw_shape(debug_image, (0, 255, 0))
                        debug_image = pred_shape[pred_idx].draw_shape(debug_image, (0, 0, 255))
                        # save the cropped image of the annotation issue for better visualization
                        gt_points = np.array(gt_shape[gt_idx].m_points)
                        crop_ul = (max(int(gt_points[0][0]) - 10, 0), max(int(gt_points[0][1]) - 10, 0))
                        crop_lr = (min(int(gt_points[0][0]) + 10, merged_image.shape[1]), min(int(gt_points[0][1]) + 10, merged_image.shape[0]))
                        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
                        cv2.imwrite(os.path.join(result_folder, f"annotation_issue_{label_name}_{gt_idx}.jpg"), debug_image[:,:,::-1])
                        label_with_issue["issue_image"] = os.path.join(result_folder, f"annotation_issue_{label_name}_{gt_idx}.jpg")
                        label_with_issue['data failure reason'] = f"Low quality annotation with IoU {iou:.2f} and Hausdorff distance {hausdorff_distance:.2f}"
                        result_info["Labels with Issue"].append(label_with_issue)

                for gt_idx in unmatched_gt_indices:
                    label_with_issue = {}
                    label_with_issue["id"] = gt_shape[gt_idx].m_attributes.get("category_id", "NA")
                    # overlay point on original image and save for results review                    
                    debug_image = merged_image.copy()
                    debug_image = gt_shape[gt_idx].draw_shape(debug_image, (0, 255, 0))
                    # save the cropped image of the unmatched annotation for better visualization
                    gt_points = np.array(gt_shape[gt_idx].m_points)
                    crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
                    crop_lr = (min(int(max(gt_points[:,0])) + 10, merged_image.shape[1]), min(int(max(gt_points[:,1])) + 10, merged_image.shape[0]))
                    debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
                    cv2.imwrite(os.path.join(result_folder, f"unmatched_annotation_{label_name}_{gt_idx}.jpg"), debug_image[:,:,::-1])
                    label_with_issue["issue_image"] = os.path.join(result_folder, f"unmatched_annotation_{label_name}_{gt_idx}.jpg")
                    label_with_issue['data failure reason'] = "Unmatched annotation, no corresponding prediction found"
                    result_info["Labels with Issue"].append(label_with_issue)
                    
                for pred_idx in unmatched_pred_indices:
                    label_with_issue = {}
                    label_with_issue["id"] = "NA"
                    # overlapy point on original image and save for results review
                    debug_image = merged_image.copy()
                    debug_image = pred_shape[pred_idx].draw_shape(debug_image, (0, 0, 255))
                    # save the cropped image of the unmatched prediction for better visualization
                    pred_points = np.array(pred_shape[pred_idx].m_points)
                    crop_ul = (max(int(min(pred_points[:,0])) - 10, 0), max(int(min(pred_points[:,1])) - 10, 0))
                    crop_lr = (min(int(max(pred_points[:,0])) + 10, merged_image.shape[1]), min(int(max(pred_points[:,1])) + 10, merged_image.shape[0]))
                    debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
                    cv2.imwrite(os.path.join(result_folder, f"unmatched_prediction_{label_name}_{pred_idx}.jpg"), debug_image[:,:,::-1])
                    label_with_issue["issue_image"] = os.path.join(result_folder, f"unmatched_prediction_{label_name}_{pred_idx}.jpg")
                    label_with_issue['data failure reason'] = "Unmatched prediction, no corresponding annotation found"
                    result_info["Labels with Issue"].append(label_with_issue) 

        # compute the performance score
        labels_evaluated = self.labels + ["InvalidLabelName"]
        insp_shapes, _, _ = read_json_data(json_data, labels_evaluated, scale=scale)
        tp, tn, fp, fn, precision, recall = compute_sam3_precison_recall_with_gt(insp_shapes, filtered_pred_shape, result_info, merged_image, result_folder, scale=scale, \
                                                                            iou_threshold=self.iou_score_threshold, \
                                                                                hausdorff_distance_threshold=self.hausdorff_distance_threshold)
        result_info["True Positives Num"] = tp
        result_info["True Negatives Num"] = tn
        result_info["False Positives Num"] = fp
        result_info["False Negatives Num"] = fn
        result_info["Precision"] = precision
        result_info["Recall"] = recall

        # save result info to json
        with open(os.path.join(result_folder, "sam3_data_validation_result.json"), "w") as f:
            json.dump(result_info, f, indent=4)

        return result_info
            
 