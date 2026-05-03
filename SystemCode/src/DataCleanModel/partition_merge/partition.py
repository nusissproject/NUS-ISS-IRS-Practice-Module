import numpy as np
from typing import Dict, Tuple, List
from src.DataCleanModel.partition_merge.bond_program_partition_base import PartitionBoxItemInfo, BoxInfo, Coord3D, itf_chc_partition


class Partition:
    def __init__(self,
                 id_idx: int,
                 center: Tuple[float, float],
                 rois: list[Dict[int, Tuple[float, float, float, float]]] = None,
                 size: Tuple[int, int] = None, # parition box size
                 offset: Tuple[float, float] = None,
                 opsize: Tuple[int, int] = None
            ):
        self.id = id_idx
        self.center = center # crop center
        self.rois = rois or [] # list of object rois
        self.size = size or [] # partition box size
        self.offset = offset or [] # crop offset
        self.opsize = opsize or [] # crop size
    
    def area(self):
        if self.size:
            return self.size[0] * self.size[1]
        else:
            return 0

        
def naive_frame_partition(frame_size:Tuple[int,int],
                          frame_roi:Tuple[float,float,float,float],
                          target_size:Tuple[int,int],
                          min_overlap_ratio:float=0.1) -> list[Partition]:
    partitions = []
    tgt_w, tgt_h = target_size
    crop_frame_offset = (frame_roi[0],frame_roi[1])
    crop_frame_size = (frame_roi[2]-frame_roi[0], frame_roi[3]-frame_roi[1])
    partition_x_num = max(int(np.ceil(crop_frame_size[0]/(tgt_w*(1 - min_overlap_ratio)))), 1)
    partition_y_num = max(int(np.ceil(crop_frame_size[1]/(tgt_h*(1 - min_overlap_ratio)))), 1)
    step_x = crop_frame_size[0]/partition_x_num
    step_y = crop_frame_size[1]/partition_y_num
    for i in range(partition_x_num):
        for j in range(partition_y_num):
            start_x = int(crop_frame_offset[0] + i*step_x)
            start_y = int(crop_frame_offset[1] + j*step_y)
            end_x = min(start_x + tgt_w, frame_size[0])
            end_y = min(start_y + tgt_h, frame_size[1])
            center_x = (start_x + end_x)/2.0
            center_y = (start_y + end_y)/2.0
            partitions.append(Partition(center=(center_x, center_y),offset=(start_x, start_y),size=(end_x-start_x, end_y-start_y), opsize=target_size))
    return partitions

# Algorithm Flow:
# Step 1: Select out those rois larger than max_roi_size
# Step 2: Use Hierarchical Clustering to cluster the remaining rois based on their centers
def device_file_frame_partition(frame_size:Tuple[int,int],
                                 object_rois:List[Tuple[float,float,float,float]],
                                 target_size:Tuple[int,int]) -> list[Partition]:
    large_rois = []
    small_rois = []
    partitions = []
    for roi_id, roi in enumerate(object_rois):
        roi = list(roi)
        roi_w = roi[2] - roi[0]
        roi_h = roi[3] - roi[1]
        if roi_w > target_size[0] or roi_h > target_size[1]:
            large_rois.append((roi_id, roi))
        else:
            small_rois.append((roi_id, roi))

    pst_input = PartitionBoxItemInfo([])
    for roi_id, roi in small_rois:
        box_info = BoxInfo(
            rco3d_ul_corner=Coord3D(roi[0], roi[1], 0),
            rco3d_lr_corner=Coord3D(roi[2], roi[3], 0),
        )
        pst_input.box_info.append(box_info)

    fov = Coord3D(target_size[0], target_size[1], 0)
    results = itf_chc_partition(pst_input, fov, None)
    id_idx = 0
    for box in results.single_boxes:
        center_x = (box.rco_ul_corner.x + box.rco_lr_corner.x)/2.0
        center_y = (box.rco_ul_corner.y + box.rco_lr_corner.y)/2.0
        rois = []
        for box_idx in box.box_index_list:
            roi_id, roi = small_rois[box_idx]
            rois.append((roi_id, roi))
        # find mininum x and y for all rois
        min_x = min(roi[0] for _, roi in rois)
        min_y = min(roi[1] for _, roi in rois)
        # find maximum x and y for all rois  
        max_x = max(roi[2] for _, roi in rois)
        max_y = max(roi[3] for _, roi in rois)
        box_size_x = max_x - min_x
        box_size_y = max_y - min_y
        partitions.append(Partition(id_idx=id_idx, center=(center_x, center_y), rois = rois, offset=(min_x, min_y), size=(box_size_x, box_size_y)))
        id_idx += 1
    return partitions


