"""
JSON writer for ATS in-house labeller
"""

import os
import json
import cv2

from shapely.geometry import Polygon
import numpy as np
from typing import Tuple,List


FIELD_MISSING_VALUE = "NA"
LABEL_SHAPE_LIST = ("circle", "ellipse", "line", "point", "polygon", "rectangle")
GOOD_STATUS_VALUE = "OK"

# TO DO: 
# 1) add constructor to with input of json file 
# 2) add properties other than label and shape to make sure file consistency

def _normalize(vector: np.ndarray) -> np.ndarray:
        """
        Normalizes the input vector.
        :param vector: Input vector as a numpy array
        :return: Normalized vector as a numpy array
        """
        return vector / np.linalg.norm(vector)

def _compute_normal(tangent: np.ndarray) -> np.ndarray:
        """
        Computes the normal of the input tangent vector.
        :param tangent: Input tangent vector as a numpy array
        :return: Normal vector as a numpy array
        """
        return np.array([tangent[1], -tangent[0]])

def _handle_miter_join(normal: np.ndarray, next_normal: np.ndarray, half_thickness: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the miter join points using input normals and half_thickness.
        :param normal: Normal vector of the current segment as a numpy array
        :param next_normal: Normal vector of the next segment as a numpy array
        :param half_thickness: Half the thickness of the current point
        :return: Left and right miter join points as numpy arrays
        """
        miter_vec = (normal + next_normal) / 2
        miter_vec /= np.linalg.norm(miter_vec)
        miter_vec *= half_thickness

        cos_theta = np.dot(normal, miter_vec) / ((np.linalg.norm(normal) * np.linalg.norm(miter_vec)) + 1e-8)
        miter_len = half_thickness / (cos_theta + 1e-8)

        if miter_len < 10 * half_thickness:  # limiting miter length to avoid extreme spikes
            left_point = miter_vec
            right_point = -miter_vec
        else:
            left_point = next_normal * half_thickness
            right_point = -next_normal * half_thickness

        return left_point, right_point

def create_thickness_polygon(points: List[Tuple[float, float, float]],
                             thickness: List[float]) -> List[Tuple[float, float]]:
        """
        Creates a polygon that represents a line through input points with the specified thickness at each point.
        :param points_with_thickness: List of tuples (x, y, t), where x and y are coordinates and t is the thickness
        :return: List of tuples (x, y) representing the coordinates of the output polygon
        """
        num_points = len(points)
        left_points = [None] * num_points
        right_points = [None] * num_points

        for i in range(num_points - 1):
            curr_pt = points[i]
            next_pt = points[i + 1]
            curr_t = thickness[i]
            next_t = thickness[i + 1]

            diff = np.array([next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]])
            tangent = _normalize(diff)
            normal = _compute_normal(tangent)

            half_thickness_curr = curr_t / 2
            half_thickness_next = next_t / 2

            if i == 0:
                left_points[i] = tuple(curr_pt + normal * half_thickness_curr)
                right_points[i] = tuple(curr_pt - normal * half_thickness_curr)

            if i < num_points - 2:
                next2_pt = points[i + 2]
                next_diff = np.array([next2_pt[0] - next_pt[0], next2_pt[1] - next_pt[1]])
                next_tangent = _normalize(next_diff)
                next_normal = _compute_normal(next_tangent)

                left_pt, right_pt = _handle_miter_join(normal, next_normal, half_thickness_next)
                left_points[i + 1] = tuple(next_pt + left_pt)
                right_points[i + 1] = tuple(next_pt + right_pt)
            else:
                left_points[i + 1] = tuple(next_pt + normal * half_thickness_next)
                right_points[i + 1] = tuple(next_pt - normal * half_thickness_next)
        # Combine the left and right points to get the outline
        outline_points = left_points + right_points[::-1]

        return outline_points

class AnnotationFileGenerator:
    def __init__(self):
        self.m_package = dict(
        customer=FIELD_MISSING_VALUE,
        name=FIELD_MISSING_VALUE
        )
        self.m_optics = dict(
            assembly_id=FIELD_MISSING_VALUE,
            resolution=FIELD_MISSING_VALUE
        )

        self.m_light = [] # array of LightInfo

        self.m_unit = [] # array of UnitInfo

        self.m_labels = [] # array of LabelInfo

        self.m_vertexColor = [0,255,0,255]

        self.m_lineColor = [0,255,0,128]

        self.m_fillColor = [255,0,0,128]

        self.m_json_ext_file = "Label_Ext.json"

        self.m_global_attributes = dict()

    def set_package_info(self, customer, name):
        self.m_package["customer"] = customer
        self.m_package["name"] = name

    def set_optics_info(self, assembly_id, resolution):
        self.m_optics["assembly_id"] = assembly_id
        self.m_optics["resolution"] = resolution

    def add_label_info(self, label_info):
        assert(type(label_info) is LabelInfo)
        self.m_labels.append(label_info)

    def add_global_attribute(self, key, value):
        self.m_global_attributes[key] = value

    def reset(self):
        self.m_package = dict(
            customer=FIELD_MISSING_VALUE,
            name=FIELD_MISSING_VALUE
        )
        self.m_optics = dict(
            assembly_id=FIELD_MISSING_VALUE,
            resolution=FIELD_MISSING_VALUE
        )

        self.m_labels = []  # array of LabelInfo

    def serialize_to_json_ext(self, dst_folder):
        data = dict(
            package=self.m_package,
            optics=self.m_optics,
            Labels=[m.serialize_to_dict_ext() for m in self.m_labels]
        )
        with open(os.path.join(dst_folder, self.m_json_ext_file), "w") as f:
            json.dump(data,f, indent=2)

    def serialize_to_json(self, dst_folder, json_file):
        data = dict(
            flags={},
            Labels=[m.serialize_to_dict() for m in self.m_labels],
            vertexColor=self.m_vertexColor,
            lineColor=self.m_lineColor,
            fillColor=self.m_fillColor,
            globalAttributes=self.m_global_attributes
        )
        # backup current file if exists
        backup_file = os.path.join(dst_folder, json_file + ".bak")
        if os.path.exists(backup_file):
            os.remove(backup_file)
        if os.path.exists(os.path.join(dst_folder, json_file)):
            os.rename(os.path.join(dst_folder, json_file), backup_file)

        with open(os.path.join(dst_folder, json_file),"w") as f:
            json.dump(data,f, indent=2)


class LabelInfo:
    def __init__(self,
                 label=FIELD_MISSING_VALUE,
                 status=GOOD_STATUS_VALUE,
                 unit_id=0):
        self.m_label = label
        self.m_status = status
        self.m_unit_id = unit_id
        self.m_shape = []

    def reset(self):
        # self.m_label = FIELD_MISSING_VALUE
        self.m_shape = []
        # self.m_status = GOOD_STATUS_VALUE
        # self.m_unit_id = 0

    def add_shape(self, s):
        assert(type(s) is ShapeInfo)
        self.m_shape.append(s)

    def serialize_to_dict(self):
        data = dict(Label=self.m_label,
                    Shapes=[s.serialize_to_dict() for s in self.m_shape],
                    Status=self.m_status,
                    Unit_ID=self.m_unit_id)
        return data

    def serialize_to_dict_ext(self):
        data = dict(Label=self.m_label,
                    Status=self.m_status,
                    Unit_ID=self.m_unit_id)
        return data


class ShapeInfo:
    def __init__(self, type):
        # assert(type in LABEL_SHAPE_LIST) # ignore for testing purpose
        self.m_type = type
        self.m_vertex_color = "default"
        self.m_line_color = "default"
        self.m_fill_color = "default"
        self.m_points = []
        self.m_proposed_points = None
        self.m_image_list = None
        self.m_thickness = None
        self.m_attributes = dict()
        self.m_score = 0.0
        self.m_label = FIELD_MISSING_VALUE

    def reset(self):
        self.m_vertex_color = "default"
        self.m_line_color = "default"
        self.m_fill_color = "default"
        self.m_type = FIELD_MISSING_VALUE
        self.m_points = []
        self.m_proposed_points = None
        self.m_thickness = None
        self.m_attributes = dict()
        self.m_image_list = None
        self.m_proposed_points = None

    # p0 and p1 are two circle points
    def set_circle_point(self,
                         center,
                         radius,
                         image_list=[],
                         attribubtes=dict()):
        assert(self.m_type == "circle")
        assert(center is not None)
        assert(radius is not None)
        assert(len(center) == 2)
        p0 = list(center)
        p1 = list(center)
        p0[0] = p0[0] - radius
        p1[0] = p1[0] + radius
        self.m_points = []
        self.m_points.append(p0)
        self.m_points.append(p1)
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    # corners are four corners of ellipse
    def set_ellipse_point(self,
                          major_axis_left_tip,
                          major_axis_right_tip,
                          minor_axis_upper_tip,
                          minor_axis_lower_tip,
                          image_list=[],
                          attribubtes=dict()):
        assert(self.m_type == "ellipse")
        assert(major_axis_left_tip is not None)
        assert(major_axis_right_tip is not None)
        assert(minor_axis_upper_tip is not None)
        assert(minor_axis_lower_tip is not None)
        assert(len(major_axis_left_tip) == 2)
        assert(len(major_axis_right_tip) == 2)
        assert(len(minor_axis_upper_tip) == 2)
        assert(len(minor_axis_lower_tip) == 2)
        self.m_points = []
        self.m_points.append(major_axis_left_tip)
        self.m_points.append(major_axis_right_tip)
        self.m_points.append(minor_axis_upper_tip)
        self.m_points.append(minor_axis_lower_tip)
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    def set_point_point(self,
                        point,
                        image_list=[],
                        attribubtes=dict()):
        assert(self.m_type == "point")
        assert(point is not None)
        assert(len(point) == 2)
        self.m_points = []
        self.m_points.append(point)
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    def set_line_point(self,
                       start,
                       end,
                       image_list=[],
                       attribubtes=dict()):
        assert(self.m_type == "line")
        assert(start is not None)
        assert(end is not None)
        assert(len(start) == 2)
        assert(len(end) == 2)
        self.m_points = []
        self.m_points.append(start)
        self.m_points.append(end)
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    def set_rectangle_point(self,
                            upper_left_corner,
                            lower_right_corner,
                            image_list=[],
                            attribubtes=dict()):
        assert(self.m_type == "rectangle")
        assert(upper_left_corner is not None)
        assert(lower_right_corner is not None)
        assert(len(upper_left_corner) == 2)
        assert(len(lower_right_corner) == 2)
        width = lower_right_corner[0] - upper_left_corner[0]
        upper_right_corner = list(upper_left_corner)
        upper_right_corner[0] = upper_right_corner[0] + width
        lower_left_corner = list(lower_right_corner)
        lower_left_corner[0] = lower_left_corner[0] - width
        self.m_points = []
        self.m_points.append(upper_left_corner)
        self.m_points.append(upper_right_corner)
        self.m_points.append(lower_right_corner)
        self.m_points.append(lower_left_corner)
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    def set_polygon_point(self,
                          point_list,
                          image_list=[],
                          attribubtes=dict()):
        assert(self.m_type == "polygon")
        assert(point_list is not None)
        assert(len(point_list) >= 2)
        self.m_points = list(point_list)
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    def set_linepolygon_point(self,
                            point_list,
                            thickness,
                            image_list=[],
                            attribubtes=dict()):
        assert(self.m_type == "polygon")
        assert(point_list is not None)
        assert(thickness is not None)
        assert(len(point_list) >= 2)
        self.m_points = list(point_list)
        self.m_thickness = thickness
        self.m_image_list = image_list
        self.m_attributes = attribubtes

    # function for general shape setting
    def set_general_shape_point(self,
                              point_list,
                              label,
                              proposed_points_list=None,
                              thickness=None,
                              image_list=None,
                              attribubtes=dict()):
        assert(point_list is not None)
        self.m_points = list(point_list)
        self.m_label = label
        self.m_proposed_points = list(proposed_points_list) if proposed_points_list is not None else None
        self.m_thickness = thickness
        self.m_image_list = image_list if image_list is not None else None
        self.m_attributes = attribubtes

    def get_polygon(self)->Polygon:
        # create shapely polygon from self.m_points, only for polygon, rectangle and ellipse shapes
        assert(self.m_type in ["polygon", "rectangle", "ellipse"])
        polygon = Polygon(self.m_points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # fixes some self-intersections
        return polygon

    def draw_shape(self, 
                   image:np.ndarray,
                   color:Tuple[int, int,int],
                   skip_type_error:bool=False)->np.ndarray:
        # draw the shape on the image, only for polygon, rectangle and ellipse shapes
        if self.m_type == "polygon" or self.m_type == "rectangle":
            gt_polygon = np.array(self.m_points, dtype=np.int32)
            if self.m_thickness is not None:
                gt_polygon = create_thickness_polygon(self.m_points, self.m_thickness)
                gt_polygon = np.array(gt_polygon, dtype=np.int32)
            cv2.polylines(image, [gt_polygon], True, color, 2)  
        elif self.m_type == "circle":
            center = (int((self.m_points[0][0] + self.m_points[1][0]) / 2), int((self.m_points[0][1] + self.m_points[1][1]) / 2))
            radius = int(((self.m_points[0][0] - self.m_points[1][0])**2 + (self.m_points[0][1] - self.m_points[1][1])**2)**0.5 / 2)
            cv2.circle(image, center, radius, color, 2)
        elif self.m_type == "ellipse":
            major_axis_left_tip = self.m_points[0]
            major_axis_right_tip = self.m_points[1]
            minor_axis_upper_tip = self.m_points[2]
            minor_axis_lower_tip = self.m_points[3]
            center = (int((major_axis_left_tip[0] + major_axis_right_tip[0]) / 2), int((minor_axis_upper_tip[1] + minor_axis_lower_tip[1]) / 2))
            major_axis_length = int(((major_axis_right_tip[0] - major_axis_left_tip[0])**2 + (major_axis_right_tip[1] - major_axis_left_tip[1])**2)**0.5)
            minor_axis_length = int(((minor_axis_upper_tip[0] - minor_axis_lower_tip[0])**2 + (minor_axis_upper_tip[1] - minor_axis_lower_tip[1])**2)**0.5)
            angle = int(np.arctan2(major_axis_right_tip[1] - major_axis_left_tip[1], major_axis_right_tip[0] - major_axis_left_tip[0]) * 180 / np.pi)
            cv2.ellipse(image, center, (major_axis_length // 2, minor_axis_length // 2), angle, 0, 360, color, 2)
        if skip_type_error and self.m_type not in ["polygon", "rectangle", "circle", "ellipse"]:
            gt_polygon = np.array(self.m_points, dtype=np.int32)
            if self.m_thickness is not None:
                gt_polygon = create_thickness_polygon(self.m_points, self.m_thickness)
                gt_polygon = np.array(gt_polygon, dtype=np.int32)
            cv2.polylines(image, [gt_polygon], True, color, 2)  
        return image
    
    def is_inside_roi(self, roi: Tuple[float, float, float, float]) -> bool:
        # check if the shape is inside the roi, only for polygon, rectangle and ellipse shapes
        if self.m_type in ["polygon", "rectangle", "ellipse"]:
            shape_polygon = self.get_polygon()
            roi_polygon = Polygon([(roi[0], roi[1]), (roi[2], roi[1]), (roi[2], roi[3]), (roi[0], roi[3])])
            return shape_polygon.within(roi_polygon)
        else:
            # for other shapes, we can check if the center point is within the roi
            center_x = sum([p[0] for p in self.m_points]) / len(self.m_points)
            center_y = sum([p[1] for p in self.m_points]) / len(self.m_points)
            return roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]
        
    def translate(self, x_offset, y_offset):
        self.m_points = [(p[0] + x_offset, p[1] + y_offset) for p in self.m_points]

    def transpose(self):
        self.m_points = [(p[1], p[0]) for p in self.m_points]
  
    def serialize_to_dict(self):
        data = dict(
            vertex_color=self.m_vertex_color,
            line_color=self.m_line_color,
            fill_color=self.m_fill_color,
            Type=self.m_type,
            points=self.m_points,
            proposed_points=self.m_proposed_points,
            Image_List=self.m_image_list,
            thickness=self.m_thickness,
            Attributes=self.m_attributes
        )
        # remove None values for better json readability
        data = {k: v for k, v in data.items() if v is not None}
        return data






