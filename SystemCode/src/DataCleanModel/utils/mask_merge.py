# Merge mask from different partitions
from typing_extensions import Tuple

from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo
from src.DataCleanModel.utils.mask_matcher import polygon_iou
from shapely.geometry import MultiPolygon
from src.DataCleanModel.utils.ai_labeler_json_processer import generate_roi

def find_largest_polygon(mpoly: MultiPolygon):
    if mpoly.is_empty:
        return None
    if mpoly.geom_type == "Polygon":
        return mpoly
    print("MultiPolygon with {} polygons, selecting the largest one".format(len(mpoly.geoms)))
    return max(mpoly.geoms, key=lambda poly: poly.area)

"""
cdef NMS(float[:, ::1] final_probs , float[:, ::1] final_bbox):
    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,class_loop,index,index2

  
    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        for index in range(pred_length):
            if final_probs[index,class_loop] == 0: continue
            for index2 in range(index+1,pred_length):
                if final_probs[index2,class_loop] == 0: continue
                if index==index2 : continue
                if box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.4:
                    if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                        final_probs[index, class_loop] =0
                        break
                    final_probs[index2,class_loop]=0
            
            if index not in indices:
                bb=BoundBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                bb.c = final_bbox[index, 4]
                bb.probs = np.asarray(final_probs[index,:])
                boxes.append(bb)
                indices.add(index)
    return boxes
"""

def mask_nms(shapes: list[ShapeInfo],
              iou_threshold: float) -> list[ShapeInfo]:
    """
    Non-maximum suppression to remove redundant shapes with large overlap.
    """
    if len(shapes) == 0:
        return []
    # sort shapes by score in descending order
    shapes = sorted(shapes, key=lambda x: x.m_score, reverse=True)
    keep_shapes = []
    indices = []
    for index in range(len(shapes)):
        if shapes[index].m_score == 0:
            continue
        for index2 in range(index+1, len(shapes)):
            if shapes[index2].m_score == 0:
                continue
            if index == index2:
                continue
            iou, _, _,_,_ = polygon_iou(shapes[index], shapes[index2])
            if iou >= iou_threshold:
                shapes[index2].m_score = 0
        if index not in indices:
            keep_shapes.append(shapes[index])
            indices.append(index)
    return keep_shapes


def remove_partial_overlapping_shapes(shapes: list[ShapeInfo],
                                   min_partial_overlapping_ratio: float) -> list[ShapeInfo]:
    if len(shapes) == 0:
        return []
    keep_shapes = []
    indices = []
    for index in range(len(shapes)):
        if shapes[index].m_score == 0:
            continue
        for index2 in range(index+1, len(shapes)):
            if index == index2:
                continue
            if shapes[index2].m_score == 0:
                continue
            iou, intersection_area, _, area1, area2 = polygon_iou(shapes[index], shapes[index2])
            if intersection_area / (area1 + 0.00001) >= min_partial_overlapping_ratio:
                # shape idnex is inside index2
                shapes[index].m_score = 0
                break
            elif intersection_area / (area2 + 0.00001) >= min_partial_overlapping_ratio:
                # shape index2 is inside index
                shapes[index2].m_score = 0
        if index not in indices:
            if shapes[index].m_score > 0.1:
                keep_shapes.append(shapes[index])
                indices.append(index)
    return keep_shapes


# PLAN
# Input:
# - list of ShapeInfo
# - min. intersection area
# Output:
# - list of ShapeInfo with merged masks
# step 1: run nms to remove redundant shapes with large overlap
# Step 2: Compute intersection area between each pair of ShapeInfo using _polygon_iou function from mask_matcher.py
# Step 3: for every shape select the one with largest intersection area, if the intersection area is larger than the threshold, merge the two shapes into one shape (can be done by taking the union of the two shapes using shapely library)
def mask_merge(shapes: list[ShapeInfo],
                nms_iou_threshold: float,
                min_intersection_area:float,
                min_partial_overlapping_ratio:float,
                boundary_mask_removed:bool) -> list[ShapeInfo]:
    # filter those nearly fully overlapped shapes by nms
    #filter_shapes = mask_nms(shapes, nms_iou_threshold)
    filter_shapes = shapes
    # filter those shapes that are nearly inside another shape by intersection area
    #for i in range(3):
    filter_shapes = remove_partial_overlapping_shapes(filter_shapes, min_partial_overlapping_ratio)

    def _mask_merege(_shapes: list[ShapeInfo]) -> list[ShapeInfo]:
        merged_shapes = []
        merged_flag = [False] * len(_shapes)
        for index in range(len(_shapes)):
            max_intersection_area = min_intersection_area
            max_intersection_shape = None
            for index2 in range(index+1, len(_shapes)):
                if merged_flag[index2]:
                    continue
                _, intersection_area, _, _, _ = polygon_iou(_shapes[index], _shapes[index2])
                if intersection_area > max_intersection_area and \
                    not merged_flag[index2]:
                    max_intersection_area = intersection_area
                    max_intersection_shape = index2
            # add condition that the intersection area must be in partition overlapping region
            if max_intersection_area > min_intersection_area and \
                max_intersection_shape is not None:
                # merge shape and max_intersection_shape into one shape by taking the union of the two shapes using shapely library
                shape_polygon = _shapes[index].get_polygon()
                other_polygon = _shapes[max_intersection_shape].get_polygon()
                merged_polygon = shape_polygon.union(other_polygon)
                #print(max_intersection_area)    
                largest_polygon = find_largest_polygon(merged_polygon)
                merged_shape = ShapeInfo(type='polygon')
                merged_shape.set_polygon_point(list(largest_polygon.exterior.coords))
                merged_shape.m_score = max(_shapes[index].m_score, _shapes[max_intersection_shape].m_score)
                merged_flag[max_intersection_shape] = True
                merged_flag[index] = True
                merged_shapes.append(merged_shape)
            else:
                merged_shapes.append(_shapes[index])
        return merged_shapes
    
    #for i in range(3):
    if not boundary_mask_removed:
        filter_shapes = _mask_merege(filter_shapes)
    return filter_shapes

def is_close_to_boundary(shape: ShapeInfo, 
                         frame_offset: Tuple[float, float], 
                         frame_size: Tuple[int, int], 
                         tolerance: float) -> bool:
    roi = generate_roi([shape], shape.m_label, 0)
    xmin,ymin,xmax,ymax = roi[0]
    left = xmin - frame_offset[0]
    upper = ymin - frame_offset[1]
    right = frame_offset[0] + frame_size[0] - xmax
    lower = frame_offset[1] + frame_size[1] - ymax
    return left < tolerance or upper < tolerance or right < tolerance or lower < tolerance
