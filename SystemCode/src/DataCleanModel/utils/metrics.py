
from shapely.geometry import Polygon
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo, create_thickness_polygon
from src.DataCleanModel.utils.ai_labeler_json_processer import generate_roi
from src.DataCleanModel.utils.mask_matcher import polygon_iou
import cv2
from typing import Dict, List, Tuple
import numpy as np
import os

def calculate_iou_between_polygons(gt_shape:ShapeInfo, 
                                   pred_shape:ShapeInfo) -> float:
    """
    Calculate the Intersection over Union (IoU) between two polygons.
    
    Returns:
    float: The IoU value between 0 and 1.
    """
    gt_polygon = Polygon(gt_shape.m_points)
    pred_polygon = Polygon(pred_shape.m_points)
    
    intersection_area = gt_polygon.intersection(pred_polygon).area
    union_area = gt_polygon.union(pred_polygon).area
    
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def calculate_hausdorff_distance(gt_shape:ShapeInfo, 
                                 pred_shape:ShapeInfo) -> float:
    """
    Calculate the Hausdorff distance between two polygons.
    
    Returns:
    float: The Hausdorff distance.
    """

    if gt_shape.m_type == "polygon" or gt_shape.m_type == "rectangle":
        gt_polygon = Polygon(gt_shape.m_points)
        if gt_shape.m_thickness is not None:
            gt_polygon = create_thickness_polygon(gt_shape.m_points, gt_shape.m_thickness)
            gt_polygon = Polygon(gt_polygon)
        if not gt_polygon.is_valid:
            gt_polygon = gt_polygon.buffer(0)  # fixes some self-intersections
    else:
        gt_polygon = None

    if pred_shape.m_type == "polygon" or pred_shape.m_type == "rectangle":
        pred_polygon = Polygon(pred_shape.m_points)
        if pred_shape.m_thickness is not None:
            pred_polygon = create_thickness_polygon(pred_shape.m_points, pred_shape.m_thickness)
            pred_polygon = Polygon(pred_polygon)
        if not pred_polygon.is_valid:
            pred_polygon = pred_polygon.buffer(0)  # fixes some self-intersections
    else:
        pred_polygon = None
    if gt_polygon is None or pred_polygon is None:
        return 0  # if either shape is not a polygon, return infinity as distance
    else:
        hausdorff_distance = gt_polygon.hausdorff_distance(pred_polygon)
        return hausdorff_distance
    

def compute_sam3_precison_recall_with_gt(inspect_shapes:List[ShapeInfo], 
                                    pred_shapes:List[ShapeInfo], 
                                    result_info:Dict,
                                    image:np.ndarray,
                                    result_folder:str,
                                    scale:float=1.0,
                                    iou_threshold=0.8,
                                    hausdorff_distance_threshold=10) -> Tuple[float, float, float, float, float, float]:
    # extract gt_shape from inspect_shape
    tp, tn, fp, fn = 0, 0, 0, 0
    result_info["False Positives"] = []
    result_info["False Negatives"] = []

    def save_fn_issue_info(gs, ps, img, iou, hd):
        fn_issue = {}
        fn_issue["id"] = gs.m_attributes.get("category_id", "NA")
        fn_issue['data failure reason'] = f"False negative prediction, no corresponding annotation found with iou {iou:.2f} and hausdorff distance {hd:.2f}"
        # overlay point and proposed point on original image and save for results review
        debug_image = img.copy()
        debug_image = gs.draw_shape(debug_image, (0, 255, 0))
        if ps is not None:
            debug_image = ps.draw_shape(debug_image, (0, 0, 255))
        # save the cropped image of the annotation issue for better visualization
        gt_points = np.array(gs.m_points)
        crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
        crop_lr = (min(int(max(gt_points[:,0])) + 10, debug_image.shape[1]), min(int(max(gt_points[:,1])) + 10, debug_image.shape[0]))
        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
        cv2.imwrite(os.path.join(result_folder, f"sam3_fn_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
        fn_issue["issue_image"] = os.path.join(result_folder, f"sam3_fn_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg")
        return fn_issue
    
    def save_fp_issue_info(gs, ps, img, iou, hd):
        fp_issue = {}
        fp_issue["id"] = gs.m_attributes.get("category_id", "NA")
        fp_issue['data failure reason'] = f"False positive prediction, no corresponding annotation found with iou {iou:.2f} and hausdorff distance {hd:.2f}"
        # overlay point and proposed point on original image and save for results review
        debug_image = img.copy()
        debug_image = gs.draw_shape(debug_image, (0, 255, 0))
        if ps is not None:
            debug_image = ps.draw_shape(debug_image, (0, 0, 255))
        # save the cropped image of the annotation issue for better visualization
        gt_points = np.array(gs.m_points)
        crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
        crop_lr = (min(int(max(gt_points[:,0])) + 10, debug_image.shape[1]), min(int(max(gt_points[:,1])) + 10, debug_image.shape[0]))
        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
        cv2.imwrite(os.path.join(result_folder, f"sam3_fp_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
        fp_issue["issue_image"] = os.path.join(result_folder, f"sam3_fp_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg")
        return fp_issue

    for s in inspect_shapes:
        if s.m_attributes.get("Label Error", None) is None:  # only consider shapes without label error attribute as gt shape
            gt_shape = s
            gt_label_error = False
        else:
            gt_points = s.m_attributes.get("Original points", [])
            # scale all the points
            gt_points = np.array(gt_points) * scale
            gt_points = gt_points.tolist()
            gt_type = s.m_attributes.get("Original Type", None)
            gt_label_error = True
            if gt_points and gt_type:
                gt_shape = ShapeInfo(type=gt_type)
                gt_shape.set_general_shape_point(gt_points, s.m_label, attribubtes=s.m_attributes)
        if gt_shape:
            if len(pred_shapes) == 0:
                if gt_label_error:
                    result_info["False Negatives"].append(save_fn_issue_info(gt_shape, None, image, 0, 0))
                    fn += 1
                else:
                    result_info["False Positives"].append(save_fp_issue_info(gt_shape, None, image, 0, 0))
                    fp += 1
                continue

            iou_matrix = np.zeros((1, len(pred_shapes)))
            for j in range(len(pred_shapes)):
                iou_matrix[0, j],_,_,_,_ = polygon_iou(gt_shape, pred_shapes[j])
            max_iou = np.max(iou_matrix)
            hausdorff_distance = calculate_hausdorff_distance(gt_shape, pred_shapes[np.argmax(iou_matrix)])
            pred_matched = max_iou > iou_threshold and hausdorff_distance < hausdorff_distance_threshold
            tp += int(pred_matched and gt_label_error)
            tn += int(pred_matched and not gt_label_error)
            fn += int(not pred_matched and gt_label_error)
            fp += int(not pred_matched and not gt_label_error)
            
            if not pred_matched and gt_label_error:
                fn_issue = save_fn_issue_info(gt_shape, pred_shapes[np.argmax(iou_matrix)], image, max_iou, hausdorff_distance)
                result_info["False Negatives"].append(fn_issue)
            if not pred_matched and not gt_label_error:
                fp_issue = save_fp_issue_info(gt_shape, pred_shapes[np.argmax(iou_matrix)], image, max_iou, hausdorff_distance)
                result_info["False Positives"].append(fp_issue)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp,tn,fp,fn,precision, recall


def compute_json_validator_precision_recall_with_gt(insp_shapes:List[ShapeInfo], 
                                                    label_with_error:Dict, 
                                                    result_info:Dict,
                                                    image:np.ndarray,
                                                    result_folder:str) -> Tuple[float, float, float, float, float, float]:
    tp, tn, fp, fn = 0, 0, 0, 0
    result_info["False Positives"] = []
    result_info["False Negatives"] = []

    def save_fn_issue_info(gs, ps, img, error_message):
        fn_issue = {}
        fn_issue["id"] = gs.m_attributes.get("category_id", "NA")
        fn_issue['data failure reason'] = error_message
        # overlay point and proposed point on original image and save for results review
        debug_image = img.copy()
        debug_image = gs.draw_shape(debug_image, (0, 255, 0))
        if ps is not None:
            debug_image = ps.draw_shape(debug_image, (0, 0, 255))
        # save the cropped image of the annotation issue for better visualization
        gt_points = np.array(gs.m_points)
        crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
        crop_lr = (min(int(max(gt_points[:,0])) + 10, debug_image.shape[1]), min(int(max(gt_points[:,1])) + 10, debug_image.shape[0]))
        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
        cv2.imwrite(os.path.join(result_folder, f"json_fn_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
        fn_issue["issue_image"] = os.path.join(result_folder, f"json_fn_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg")
        return fn_issue
    
    def save_fp_issue_info(gs, ps, img, error_message):
        fp_issue = {}
        fp_issue["id"] = gs.m_attributes.get("category_id", "NA")
        fp_issue['data failure reason'] = error_message
        # overlay point and proposed point on original image and save for results review
        debug_image = img.copy()
        debug_image = gs.draw_shape(debug_image, (0, 255, 0))
        if ps is not None:
            debug_image = ps.draw_shape(debug_image, (0, 0, 255))
        # save the cropped image of the annotation issue for better visualization
        gt_points = np.array(gs.m_points)
        crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
        crop_lr = (min(int(max(gt_points[:,0])) + 10, debug_image.shape[1]), min(int(max(gt_points[:,1])) + 10, debug_image.shape[0]))
        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
        cv2.imwrite(os.path.join(result_folder, f"json_fp_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
        fp_issue["issue_image"] = os.path.join(result_folder, f"json_fp_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg")
        return fp_issue

    label_with_error_by_id = {}
    for x in label_with_error:
        label_with_error_by_id[x.get("id", "NA")] = x.get("data failure reason", "NA")
    for s in insp_shapes:
        gt_label_error = False
        pred_label_error = False
        if s.m_attributes.get("Label Error", None) is not None:
            gt_label_error = True
        if s.m_attributes.get("category_id", -1) in label_with_error_by_id:
            pred_label_error = True
        if gt_label_error and pred_label_error:
            tp += 1
        if not gt_label_error and not pred_label_error:
            tn += 1
        if gt_label_error and not pred_label_error:
            fn += 1
            fn_issue = save_fn_issue_info(s, None, image, label_with_error_by_id.get(s.m_attributes.get("category_id", -1), "NA"))
            result_info["False Negatives"].append(fn_issue)
        if not gt_label_error and pred_label_error:
            fp += 1
            fp_issue = save_fp_issue_info(s, None, image, label_with_error_by_id.get(s.m_attributes.get("category_id", -1), "NA"))
            result_info["False Positives"].append(fp_issue)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp,tn,fp,fn,precision, recall


def compute_feature_validator_precision_recall_with_gt(insp_shapes:List[ShapeInfo], 
                                                    label_with_error:Dict, 
                                                    result_info:Dict,
                                                    image:np.ndarray,
                                                    result_folder:str) -> Tuple[float, float, float, float, float, float]:
    tp, tn, fp, fn = 0, 0, 0, 0
    result_info["False Positives"] = []
    result_info["False Negatives"] = []

    def save_fn_issue_info(gs, ps, img, error_message):
        fn_issue = {}
        fn_issue["id"] = gs.m_attributes.get("category_id", "NA")
        fn_issue['data failure reason'] = error_message
        # overlay point and proposed point on original image and save for results review
        debug_image = img.copy()
        debug_image = gs.draw_shape(debug_image, (0, 255, 0))
        if ps is not None:
            debug_image = ps.draw_shape(debug_image, (0, 0, 255))
        # save the cropped image of the annotation issue for better visualization
        gt_points = np.array(gs.m_points)
        crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
        crop_lr = (min(int(max(gt_points[:,0])) + 10, debug_image.shape[1]), min(int(max(gt_points[:,1])) + 10, debug_image.shape[0]))
        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
        cv2.imwrite(os.path.join(result_folder, f"feature_fn_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
        fn_issue["issue_image"] = os.path.join(result_folder, f"feature_fn_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg")
        return fn_issue
    
    def save_fp_issue_info(gs, ps, img, error_message):
        fp_issue = {}
        fp_issue["id"] = gs.m_attributes.get("category_id", "NA")
        fp_issue['data failure reason'] = error_message
        # overlay point and proposed point on original image and save for results review
        debug_image = img.copy()
        debug_image = gs.draw_shape(debug_image, (0, 255, 0))
        if ps is not None:
            debug_image = ps.draw_shape(debug_image, (0, 0, 255))
        # save the cropped image of the annotation issue for better visualization
        gt_points = np.array(gs.m_points)
        crop_ul = (max(int(min(gt_points[:,0])) - 10, 0), max(int(min(gt_points[:,1])) - 10, 0))
        crop_lr = (min(int(max(gt_points[:,0])) + 10, debug_image.shape[1]), min(int(max(gt_points[:,1])) + 10, debug_image.shape[0]))
        debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
        cv2.imwrite(os.path.join(result_folder, f"feature_fp_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
        fp_issue["issue_image"] = os.path.join(result_folder, f"feature_fp_issue_{gs.m_label}_{gs.m_attributes.get('category_id', 'NA')}.jpg")
        return fp_issue

    label_with_error_by_id = {}
    for x in label_with_error:
        label_with_error_by_id[x.get("id", "NA")] = x.get("data failure reason", "NA")
    for s in insp_shapes:
        gt_label_error = False
        pred_label_error = False
        if s.m_attributes.get("Label Error", None) is not None:
            gt_label_error = True
        if s.m_attributes.get("category_id", -1) in label_with_error_by_id:
            pred_label_error = True
        if gt_label_error and pred_label_error:
            tp += 1
        if not gt_label_error and not pred_label_error:
            tn += 1
        if gt_label_error and not pred_label_error:
            fn += 1
            fn_issue = save_fn_issue_info(s, None, image, "Feature Validator Error")
            result_info["False Negatives"].append(fn_issue)
        if not gt_label_error and pred_label_error:
            fp += 1
            fp_issue = save_fp_issue_info(s, None, image, "Feature Validator Error")
            result_info["False Positives"].append(fp_issue)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return tp,tn,fp,fn,precision, recall


# Compute the performance of hybrid results if any of validator predicts a shape as label error, we consider it as label error in hybrid results. Then we compute the precision and recall of the hybrid results with gt. This can reflect the upper bound performance if we consider any shape that is predicted as label error by either validator as label error and send it for human review, which can be useful for understanding how many issues may require human review and how many of them can be correctly identified by at least one of the validators.
def compute_hybrid_precision_recall_with_gt(insp_shapes:List[ShapeInfo], 
                                            result_info:List[Dict], 
                                            merged_result_info:Dict,
                                            image:np.ndarray,
                                            result_folder:str) -> Tuple[float, float, float, float, float, float]:
    # for shapes that are predicted as label error by both json validator and feature validator, we consider them as true positive if they are indeed label error in gt, otherwise false positive. For shapes that are not predicted as label error by both json validator and feature validator, we consider them as true negative if they are indeed not label error in gt, otherwise false negative. For shapes that are predicted as label error by one of the two validators but not the other, we will not consider them in the evaluation since they are not consistent between two validators and may require further human review.
    tp, tn, fp, fn = 0, 0, 0, 0
    positives_count = result_info[0].get("True Positives Num", 0) + result_info[0].get("False Negatives Num", 0)
    negatives_count = result_info[0].get("True Negatives Num", 0) + result_info[0].get("False Positives Num", 0)
    merged_result_info["False Positives"] = []
    merged_result_info["False Negatives"] = []

    # merged FP is the union all of results
    # mereged FN is the intersection of results
    merged_label_with_error = {}
    count = 1
    fn_list = []
    for x in result_info:
        for fp_issue in x.get("False Positives", []):
            id = fp_issue.get("id", "NA")
            if id != "NA":
                if id not in merged_label_with_error:
                    merged_label_with_error[id] = {"data failure reason": [fp_issue.get("data failure reason", "NA")], \
                                                   "issue_image": fp_issue.get("issue_image", "NA")}
                else:
                    merged_label_with_error[id]["data failure reason"].append(fp_issue.get("data failure reason", "NA"))
            else: # fp label not match with any annotation
                id = f"NA_{count}"
                count += 1
                merged_label_with_error[id] = {"data failure reason": [fp_issue.get("data failure reason", "NA")],\
                                                "issue_image": fp_issue.get("issue_image", "NA")}
        id_list = []
        for fn_issue in x.get("False Negatives", []):
            id_list.append(fn_issue.get("id", "NA"))
        fn_list.append(id_list)

    # write FP to merged result info
    for id, info in merged_label_with_error.items():
        fp_issue = {}
        fp_issue["id"] = id
        fp_issue['data failure reason'] = "NA"
        # for issue image, we will save all the images for this issue and let human review to check
        fp_issue["issue_image"] = info["issue_image"]
        merged_result_info["False Positives"].append(fp_issue)
        fp += 1

    # find interaction in fn_list
    if len(fn_list) > 0:
        merged_fn_id = set(fn_list[0])
        for id_list in fn_list[1:]:
            merged_fn_id = merged_fn_id.intersection(set(id_list))
    else:
        merged_fn_id = set()

    merged_label_with_error = {}        
    for x in result_info:
        for fn_issue in x.get("False Negatives", []):
            id = fn_issue.get("id", "NA")
            if id in merged_fn_id:
                if id not in merged_label_with_error:
                    merged_label_with_error[id] = {"data failure reason": [fn_issue.get("data failure reason", "NA")],\
                                                   "issue_image": fn_issue.get("issue_image", "NA")}
                else:                    
                    merged_label_with_error[id]["data failure reason"].append(fn_issue.get("data failure reason", "NA"))
    # write FN to merged result info
    for id, info in merged_label_with_error.items():
        fn_issue = {}
        fn_issue["id"] = id
        fn_issue['data failure reason'] = "NA"
        # for issue image, we will save all the images for this issue and let human review to check
        fn_issue["issue_image"] = info["issue_image"]
        merged_result_info["False Negatives"].append(fn_issue)
        fn += 1

    # write to merged result info
    tp = max(0, positives_count - fn)
    tn = max(0, negatives_count - fp)
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    return tp,tn,fp,fn,precision, recall
