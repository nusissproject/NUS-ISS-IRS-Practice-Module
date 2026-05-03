from cv2.gapi import mask
import numpy as np
import math
from typing import List, Optional,Dict, Tuple
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo, create_thickness_polygon
import cv2
import json
from skimage.measure import CircleModel


def read_json_data(json_data:Dict, 
                    label_name_list:List[str]=None,
                    scale:float=1.0)->Tuple[List[ShapeInfo], \
                                                        List[ShapeInfo], \
                                                        List[str]]:
        json_array = []
        json_array_remain = []
        all_label_name = []
        object_rois = []
        def save_to_shapeinfo(flag:bool):
            for j in range(len((json_data['Labels'][i]['Shapes']))):
                points = json_data['Labels'][i]['Shapes'][j]['points']
                proposed_points = json_data['Labels'][i]['Shapes'][j].get("proposed_points", None)
                if scale != 1.0:
                    points = [(point[0]*scale, point[1]*scale) for point in points]
                    if proposed_points is not None:
                        proposed_points = [(point[0]*scale, point[1]*scale) for point in proposed_points]
                thickness = None
                if 'thickness' in json_data['Labels'][i]['Shapes'][j]:
                    thickness = json_data['Labels'][i]['Shapes'][j]['thickness']
                    if len(thickness) > 0 and thickness[0] == 0:
                        thickness = None
                    elif len(thickness) != len(json_data['Labels'][i]['Shapes'][j]['points']):
                        thickness = None
                    s = ShapeInfo(type=json_data['Labels'][i]['Shapes'][j]['Type'])
                    s.set_general_shape_point(points, 
                                                  json_data['Labels'][i]['Label'],
                                                  proposed_points,
                                                  [t * scale for t in thickness] if thickness is not None else None, 
                                                  json_data['Labels'][i]['Shapes'][j].get('Image_List', None), 
                                                  json_data['Labels'][i]['Shapes'][j].get('Attributes', {}))
                else:
                    s = ShapeInfo(type=json_data['Labels'][i]['Shapes'][j]['Type'])
                    s.set_general_shape_point(points, 
                                                  json_data['Labels'][i]['Label'],
                                                  proposed_points, 
                                                  None, 
                                                  json_data['Labels'][i]['Shapes'][j].get('Image_List', None), 
                                                  json_data['Labels'][i]['Shapes'][j].get('Attributes', {}))
                if flag:
                    json_array.append(s)
                else:
                    json_array_remain.append(s)
                all_label_name.append(json_data['Labels'][i]['Label'])
                
        for i in range(len(json_data['Labels'])):
            for l_n in range(len(label_name_list)):
                if label_name_list is None or label_name_list[l_n].lower() in json_data['Labels'][i]['Label'].lower():
                    save_to_shapeinfo(True)
                else:
                    save_to_shapeinfo(False)
        # find uniques labels in all_label_name
        unique_label_name = set(all_label_name)
        return json_array, json_array_remain, unique_label_name


def calculate_polygon_roi(point_list: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box of a polygon defined by a list of points.
        :param point_list: List of tuples (x, y) representing the vertices of the polygon
        :return: Tuple (x_min, y_min, x_max, y_max) representing the bounding box
        """
        x_coords = [point[0] for point in point_list]
        y_coords = [point[1] for point in point_list]

        x_min = int(min(x_coords) + 0.5)
        y_min = int(min(y_coords) + 0.5)
        x_max = int(max(x_coords) + 0.5)
        y_max = int(max(y_coords) + 0.5)

        return x_min, y_min, x_max, y_max


def _calculate_circle_roi(point_list: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box of a circle defined by two points (diameter endpoints).
        :param point_list: List of two tuples (x, y) representing the endpoints of the diameter
        :return: Tuple (x_min, y_min, x_max, y_max) representing the bounding box
        """
        x1, y1 = point_list[0]
        x2, y2 = point_list[1]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        radius = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2

        x_min = int(mid_x - radius + 0.5)
        y_min = int(mid_y - radius + 0.5)
        x_max = int(mid_x + radius + 0.5)
        y_max = int(mid_y + radius + 0.5)

        return x_min, y_min, x_max, y_max


def _calculate_ellipse_roi(point_list: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box of an ellipse defined by four points.
        :param point_list: List of four tuples (x, y) representing points on the ellipse
        :return: Tuple (x_min, y_min, x_max, y_max) representing the bounding box
        """
        x_coords = [point[0] for point in point_list]
        y_coords = [point[1] for point in point_list]

        x_min = int(min(x_coords) + 0.5)
        y_min = int(min(y_coords) + 0.5)
        x_max = int(max(x_coords) + 0.5)
        y_max = int(max(y_coords) + 0.5)

        return x_min, y_min, x_max, y_max


def generate_instance_mask(mask, 
                           points: list[Tuple[float, float]], 
                           type:str, 
                           label_class: int,
                           thickness : list[float] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        roi = None
        if type == 'polygon':
            # rounding the points
            if thickness is None:
                point_array = np.array(points, dtype=np.float32) + 0.5
                cv2.fillPoly(mask, point_array.astype(np.int32), label_class)
                roi = calculate_polygon_roi(point_array)
            else:
                # draw line polygon with thickness
                point_array = create_thickness_polygon(points, thickness)
                cv2.fillPoly(mask, (np.array([point_array], dtype=np.int32) + 0.5).astype(np.int32), label_class)
                roi = calculate_polygon_roi(point_array)
        elif type == 'circle':
            x = points[0][0]
            y = points[0][1]
            x1 = points[1][0]
            y1 = points[1][1]
            mid_x = int(((x + x1) / 2) + 0.5)
            mid_y = int(((y + y1) / 2) + 0.5)
            radius = int(((math.sqrt(pow((x1 - x), 2) + pow((y1 - y), 2)) / 2) + 0.5))
            cv2.circle(mask, (mid_x, mid_y), radius, color=label_class, thickness=-1)
            roi = _calculate_circle_roi(points)
        elif type == 'ellipse':
            p1_x = points[0][0]
            p1_y = points[0][1]
            p2_x = points[1][0]
            p2_y = points[1][1]
            p3_x = points[2][0]
            p3_y = points[2][1]
            p4_x = points[3][0]
            p4_y = points[3][1]
            e_center_x = int(((p1_x + p2_x) / 2) + 0.5)
            e_center_y = int(((p1_y + p2_y) / 2) + 0.5)
            r_axis_1 = int(((math.sqrt(pow((p2_x - p1_x), 2) + pow((p2_y - p1_y), 2)) / 2) + 0.5))
            r_axis_2 = int(((math.sqrt(pow((p4_x - p3_x), 2) + pow((p4_y - p3_y), 2)) / 2) + 0.5))
            if r_axis_1 > r_axis_2:
                r_major_axis = r_axis_1
                r_minor_axis = r_axis_2
                base_pt_x = p1_x
                base_pt_y = p1_y
            else:
                r_major_axis = r_axis_2
                r_minor_axis = r_axis_1
                base_pt_x = p3_x
                base_pt_y = p3_y
                pt0_x = e_center_x + 1
                pt0_y = e_center_y
                a = int((math.sqrt(pow((base_pt_x - e_center_x), 2) + pow((base_pt_y - e_center_y), 2))))
                b = int((math.sqrt(pow((pt0_x - e_center_x), 2) + pow((pt0_y - e_center_y), 2))))
                c = int((math.sqrt(pow((base_pt_x - pt0_x), 2) + pow((base_pt_y - pt0_y), 2))))
                if (pt0_x == base_pt_x) and (pt0_y == base_pt_y):
                    angle = 0
                else:
                    radian = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                    angle = radian / np.pi * 180
                    cv2.ellipse(mask, (int(e_center_x), int(e_center_y)), (int(r_major_axis), int(r_minor_axis)),
                                    angle=-angle, startAngle=0, endAngle=360, color=label_class, thickness=-1)
            roi = _calculate_ellipse_roi(points)
        elif type == 'rectangle':
            x_min = int(np.amin(points, axis=0)[0] + 0.5)
            y_min = int(np.amin(points, axis=0)[1] + 0.5)
            x_max = int(np.amax(points, axis=0)[0] + 0.5)
            y_max = int(np.amax(points, axis=0)[1] + 0.5)
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), label_class, -1)
            roi = (x_min, y_min, x_max, y_max)
        return mask, roi

# Generate Shape Mask from Json Data Points
def generate_mask( json_data:list[ShapeInfo], 
                   mask:np.ndarray, 
                   label_name:str, 
                   label_class:int):
        rois = []
        for i in range(len(json_data)):
            if json_data[i].m_type is not None:
                if label_name.lower() in json_data[i].m_label.lower():
                    mask, roi = generate_instance_mask(
                        mask, 
                        json_data[i].m_points, 
                        json_data[i].m_type, 
                        label_class,
                        json_data[i].m_thickness if json_data[i].m_thickness is not None else None
                    )
                    rois.append(roi)
        return mask, rois


def generate_roi(json_data:List[ShapeInfo],
                 lable_name:str,
                 shift:float)->list[Tuple[int,int,int,int]]:
        rois = []
        for s in json_data:
            if s.m_type == 'polygon':
                # rounding the points
                if s.m_thickness is None:
                    point_array = np.array(s.m_points, dtype=np.float32) + 0.5
                    roi = calculate_polygon_roi(point_array)
                else:
                    # draw line polygon with thickness
                    point_array = create_thickness_polygon(s.m_points, s.m_thickness)
                    roi = calculate_polygon_roi(point_array)
            elif s.m_type == 'circle':
                roi = _calculate_circle_roi(s.m_points)
            elif s.m_type == 'ellipse':
                roi = _calculate_ellipse_roi(s.m_points)
            elif s.m_type == 'rectangle':
                x_min = int(np.amin(s.m_points, axis=0)[0] + 0.5)
                y_min = int(np.amin(s.m_points, axis=0)[1] + 0.5)
                x_max = int(np.amax(s.m_points, axis=0)[0] + 0.5)
                y_max = int(np.amax(s.m_points, axis=0)[1] + 0.5)
                roi = (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = roi
            roi = (x_min - shift, y_min - shift, x_max + shift, y_max + shift)
            rois.append(roi)
        return rois


def find_the_most_frequent_shape(json_array:list[ShapeInfo], label_name:str) -> Optional[str]:
        shape_count = {}
        for item in json_array:
            if label_name.lower() in item.m_label.lower():
                shape_type = item.m_type
                if shape_type in shape_count:
                    shape_count[shape_type] += 1
                else:
                    shape_count[shape_type] = 1

        if not shape_count:
            return None

        most_frequent_shape = max(shape_count, key=shape_count.get)
        return most_frequent_shape


def approximate_mask_with_shape(mask: np.ndarray, 
                                offset,
                                shape_type: str,
                                contour_approximation_epsilon:float) -> Tuple[str, np.ndarray]:
    # find the largest blob in the image and approximate the blob using polygon
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    if shape_type in "circle":
        circle_model = CircleModel()
        if circle_model.estimate(largest_contour.reshape(-1, 2)):
            center_x, center_y, radius = circle_model.params
            center_x += offset[0]
            center_y += offset[1]
            return "circle",np.array([[center_x - radius, center_y], [center_x + radius, center_y]])
    elif shape_type in "rectangle":
         # find the smallest enclosing rectangle of the largest contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return "rectangle", box + np.array(offset)
    # by default use polygon to approximate the shape
    approx_polygon = cv2.approxPolyDP(largest_contour, contour_approximation_epsilon, True)
    # add offset to the polygon points
    approx_polygon = approx_polygon + np.array(offset)
    # covert approx_polygon to list of list
    approx_polygon = approx_polygon.reshape(-1, 2).tolist()
    return "polygon", approx_polygon
        