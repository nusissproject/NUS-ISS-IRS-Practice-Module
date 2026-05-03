# Compare between GT mask and predicted mask using mask matcher
import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple
from shapely.geometry import Polygon, Point
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo, create_thickness_polygon
from shapely.affinity import scale, rotate

def polygon_iou(gt_shape: ShapeInfo, 
                 pred_shape: ShapeInfo) -> float:
    """
    poly*_pts: list of (x, y) corners in order (clockwise or counterclockwise)
    """
    if gt_shape.m_type not in ["polygon", "rectangle", "circle", "ellipse"] or \
        pred_shape.m_type not in ["polygon", "rectangle", "circle", "ellipse"]:
        return 0.0,0.0,0.0,0.0,0.0  # if either shape is not a polygon/rectangle/circle/ellipse, return 0 IoU and 0 areas
    if len(gt_shape.m_points) < 4 or len(pred_shape.m_points) < 4:
        return 0.0,0.0,0.0,0.0,0.0
    def _get_shape_buffer(shape:ShapeInfo):
        if shape.m_type == "polygon":
            p1 = Polygon(shape.m_points)
            if shape.m_thickness is not None:
                gt_polygon = create_thickness_polygon(shape.m_points, shape.m_thickness)
                p1 = Polygon(gt_polygon)
            if not p1.is_valid:
                p1 = p1.buffer(0)  # fixes some self-intersections
        elif shape.m_type == "rectangle":
            p1 = Polygon(shape.m_points)
            if not p1.is_valid:
                p1 = p1.buffer(0)
        elif shape.m_type == "circle":
            center = [(shape.m_points[0][0] + shape.m_points[1][0]) / 2,
                    (shape.m_points[0][1] + shape.m_points[1][1]) / 2]
            radius = ((shape.m_points[1][0] - shape.m_points[0][0]) ** 2 + \
                (shape.m_points[1][1] - shape.m_points[0][1]) ** 2) **0.5/2.0
            p1 = Point(center).buffer(radius)
        elif shape.m_type == "ellipse":
            major_axis_left_tip = shape.m_points[0]
            major_axis_right_tip = shape.m_points[1]
            minor_axis_upper_tip = shape.m_points[2]
            minor_axis_lower_tip = shape.m_points[3]
            center = [(major_axis_left_tip[0] + major_axis_right_tip[0]) / 2,
                    (minor_axis_upper_tip[1] + minor_axis_lower_tip[1]) / 2]
            major_axis_length = np.sqrt((major_axis_right_tip[0] - major_axis_left_tip[0])**2 +
                                        (major_axis_right_tip[1] - major_axis_left_tip[1])**2)
            minor_axis_length = np.sqrt((minor_axis_upper_tip[0] - minor_axis_lower_tip[0])**2 +
                                        (minor_axis_upper_tip[1] - minor_axis_lower_tip[1])**2)
            # calculate ellipse major axis angle
            angle = np.arctan2(major_axis_right_tip[1] - major_axis_left_tip[1],
                               major_axis_right_tip[0] - major_axis_left_tip[0])
            p1 = Point(center).buffer(1)
            p1 = scale(p1, xfact = major_axis_length / 2, yfact = minor_axis_length / 2)
            p1 = rotate(p1, angle * 180 / np.pi)
        return p1
    
    p1 = _get_shape_buffer(gt_shape)
    p2 = _get_shape_buffer(pred_shape)

    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    #if inter > 0:
    #    print(f"IOU between GT shape and Pred shape: {inter/union if union > 0 else 0.0}, Intersection area: {inter}, Union area: {union}, GT area: {p1.area}, Pred area: {p2.area}")
    #    print(f"GT shape points: {gt_shape.m_points}, Pred shape points: {pred_shape.m_points}")
    return 0.0 if union == 0 else inter / union, inter, union, p1.area, p2.area

def mask_matcher(gt_shape: list[ShapeInfo],
                 pred_shape: list[ShapeInfo],
                 iou_threshold=0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Compare ground truth mask and predicted mask, return matching metrics.

    Args:
        gt_mask (np.ndarray): Ground truth binary mask.
        pred_mask (np.ndarray): Predicted binary mask.

    Returns:
        list[int, int]: List of matched shape indices between gt_shape and pred_shape.
        list[int]: List of unmatched gt_shape indices.
        list[int]: List of unmatched pred_shape indices.
    """
    gt_num = len(gt_shape)
    pred_num = len(pred_shape)
    iou_matrix = np.zeros((gt_num, pred_num))
    for i in range(gt_num):
        for j in range(pred_num):
            iou_matrix[i, j],_,_,_,_ = polygon_iou(gt_shape[i], pred_shape[j])

    matched_indices = []
    unmatched_gt_indices = list(range(gt_num))
    unmatched_pred_indices = list(range(pred_num))

    for i in range(gt_num):
        for j in range(pred_num):
            if iou_matrix[i, j] >= iou_threshold:
                matched_indices.append((i, j, iou_matrix[i, j]))
                if i in unmatched_gt_indices:
                    unmatched_gt_indices.remove(i)
                if j in unmatched_pred_indices:
                    unmatched_pred_indices.remove(j)
    return matched_indices, unmatched_gt_indices, unmatched_pred_indices

    # visualize matching and unmatched shapes on image for debugging
    # matched shapes in green, unmatched gt shapes in red, unmatched pred shapes in blue
    # for debugging, we can draw the shapes on a blank image
    # debug_image = np.zeros((5000, 5000, 3), dtype=np.uint8)
    # for i, j, iou in matched_indices:
    #     color = (0, 255, 0)  # green for matched
    #     gt_shape[i].draw_shape(debug_image, color)
    #     color = (0, 255,255)  # cyan for matched pred shape
    #     pred_shape[j].draw_shape(debug_image, color)
    #     # print iou on image
    #     cv2.putText(debug_image, f"{iou:.2f}", (int(gt_shape[i].m_points[0][0]), int(gt_shape[i].m_points[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # for i in unmatched_gt_indices:
    #     color = (0, 0, 255)  # red for unmatched gt
    #     gt_shape[i].draw_shape(debug_image, color)
    # for j in unmatched_pred_indices:
    #     color = (255, 0, 0)  # blue for unmatched pred
    #     pred_shape[j].draw_shape(debug_image, color)
    # cv2.imwrite("mask_matching_debug.jpg", debug_image)

    # # visualize gt shape on image    
    # debug_image = np.zeros((5000, 5000, 3), dtype=np.uint8)
    # for i in range(gt_num):
    #     color = (0, 255, 0)  # green for gt     
    #     gt_shape[i].draw_shape(debug_image, color)
    # cv2.imwrite("gt_shapes.jpg", debug_image)

    # debug_image = np.zeros((5000, 5000, 3), dtype=np.uint8)
    # for j in range(pred_num):
    #     color = (0, 0, 255)  # red for pred     
    #     pred_shape[j].draw_shape(debug_image, color)
    # cv2.imwrite("pred_shapes.jpg", debug_image)

    

