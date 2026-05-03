from typing import Dict, List, Tuple
import numpy as np
import cv2
from copy import deepcopy

# generate annotation file with invalid label name
def generate_invalid_label_name_error(img:np.ndarray, label_json:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    labels = label_json.get("Labels", [])
    for label in labels:
        label["Label"] = "InvalidLabelName"
        for shape in label.get("Shapes", []):
            total_label_count["total"] += 1
            if "Label Error" not in shape["Attributes"]:
                shape["Attributes"]["Label Error"] = []
            shape["Attributes"]["Label Error"].append(f"Invalid label name")
            shape["Attributes"]["Original points"] = shape["points"]
            shape["Attributes"]["Original Type"] = shape["Type"]
            total_label_count["invalid label"] += 1
    return img, label_json

# gaussian blur the whole image 
def generate_blurred_image_error(img:np.ndarray, label_json:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    blurred_img = cv2.GaussianBlur(img, (15, 15), 30)
    labels = label_json.get("Labels", [])
    for label in labels:
        for shape in label.get("Shapes", []):
            total_label_count["total"] += 1
            if "Label Error" not in shape["Attributes"]:
                shape["Attributes"]["Label Error"] = []
            shape["Attributes"]["Label Error"].append("Image is blurred")
            shape["Attributes"]["Original points"] = shape["points"]
            shape["Attributes"]["Original Type"] = shape["Type"]
            total_label_count["blur"] += 1
    return blurred_img, label_json

# low contrast the whole image
def generate_low_contrast_image_error(img:np.ndarray, label_json:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    img = img.astype(np.float32)
    # random generate a contrast factor between 0.0 and 0.1
    contrast_factor = np.random.uniform(0.0, 0.1)
    img = (img - img.min()) * contrast_factor + img.min()
    img = img.astype(np.uint8)
    labels = label_json.get("Labels", [])
    for label in labels:
        for shape in label.get("Shapes", []):
            total_label_count["total"] += 1
            if "Label Error" not in shape["Attributes"]:
                shape["Attributes"]["Label Error"] = []
            shape["Attributes"]["Label Error"].append("Image has low contrast")
            shape["Attributes"]["Original points"] = shape["points"]
            shape["Attributes"]["Original Type"] = shape["Type"]
            total_label_count["low_contrast"] += 1
    return img, label_json


# darken the whole image
def generate_darkened_image_error(img:np.ndarray, label_json:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    img = img.astype(np.float32)
    dark_factor = np.random.uniform(0.0, 0.1)
    img = (img * dark_factor).astype(np.uint8)
    labels = label_json.get("Labels", [])

    for label in labels:
        for shape in label.get("Shapes", []):
            total_label_count["total"] += 1
            if "Label Error" not in shape["Attributes"]:
                shape["Attributes"]["Label Error"] = []
            shape["Attributes"]["Label Error"].append("Image is darkened")
            shape["Attributes"]["Original points"] = shape["points"]
            shape["Attributes"]["Original Type"] = shape["Type"]
            total_label_count["darken"] += 1
    return img, label_json


def generate_invalid_shape_type_error(img:np.ndarray, shape:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    ori_type = shape["Type"]
    shape["Type"] = "InvalidShapeType"
    if "Label Error" not in shape["Attributes"]:
        shape["Attributes"]["Label Error"] = []
    shape["Attributes"]["Label Error"].append(f"Invalid shape type {shape.get('Type', 'NA')}")
    shape["Attributes"]["Original Type"] = ori_type
    shape["Attributes"]["Original points"] = shape["points"]
    total_label_count["total"] += 1
    total_label_count["invalid shape type"] += 1
    return img, shape

def generate_out_of_range_points_error(img:np.ndarray, shape:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    points = deepcopy(shape["points"])
    shape["points"].append([6000, 6000])  # Add a point that is out of range
    if "Label Error" not in shape["Attributes"]:
        shape["Attributes"]["Label Error"] = []
    shape["Attributes"]["Label Error"].append("Out of range points detected")
    shape["Attributes"]["Original points"] = points
    shape["Attributes"]["Original Type"] = shape["Type"]
    total_label_count["total"] += 1
    total_label_count["out of range points"] += 1
    return img, shape

def generate_annotation_by_rotation(img:np.ndarray, shape:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    points = deepcopy(shape["points"])
    center = np.mean(points, axis=0)
    # generate angle between 5 and 10 degree
    angle = np.random.uniform(5, 10)
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    rotated_points = np.dot(points - center, rotation_matrix) + center
    rotated_points[:, 0] = np.clip(rotated_points[:, 0], 0, img.shape[1]-1)
    rotated_points[:, 1] = np.clip(rotated_points[:, 1], 0, img.shape[0]-1)
    shape["points"] = rotated_points.tolist()
    if "Label Error" not in shape["Attributes"]:
        shape["Attributes"]["Label Error"] = []
    shape["Attributes"]["Label Error"].append(f"Shape points are rotated by {angle:.2f} degrees")
    shape["Attributes"]["Original points"] = points
    shape["Attributes"]["Original Type"] = shape["Type"]
    total_label_count["total"] += 1
    total_label_count["rotation"] += 1
    return img, shape

def generate_annotation_by_scaling(img:np.ndarray, shape:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    points = deepcopy(shape["points"])
    center = np.mean(points, axis=0)
    # generate scale factor between 1.3 and 1.6
    scale_factor = np.random.uniform(1.3, 1.6)
    scaled_points = (points - center) * scale_factor + center
    # clip the points to be within the image size width and height
    scaled_points[:, 0] = np.clip(scaled_points[:, 0], 0, img.shape[1]-1)
    scaled_points[:, 1] = np.clip(scaled_points[:, 1], 0, img.shape[0]-1)
    shape["points"] = scaled_points.tolist()
    if "Label Error" not in shape["Attributes"]:
        shape["Attributes"]["Label Error"] = []
    shape["Attributes"]["Label Error"].append(f"Shape points are scaled by a factor of {scale_factor:.2f}")
    shape["Attributes"]["Original points"] = points
    shape["Attributes"]["Original Type"] = shape["Type"]
    total_label_count["total"] += 1
    total_label_count["scaling"] += 1
    return img, shape


WHOLE_IMAGE_ERROR_PROB = 0.1
SHAPE_ERROR_PROB = 0.4

def generate_synthetic_data_with_error(img:np.ndarray, label_json:Dict, total_label_count:Dict)->Tuple[np.ndarray, Dict]:
    # generate whole image error with a certain probability
    if np.random.rand() < WHOLE_IMAGE_ERROR_PROB:
        error_type = np.random.choice(["invalid label","blur", "low_contrast", "darken"])
        if error_type == "blur":
            noisy_image, noisy_label = generate_blurred_image_error(img, label_json, total_label_count)
            return noisy_image, noisy_label
        elif error_type == "low_contrast":
            noisy_image, noisy_label = generate_low_contrast_image_error(img, label_json, total_label_count)
            return noisy_image, noisy_label
        elif error_type == "darken":
            noisy_image, noisy_label = generate_darkened_image_error(img, label_json, total_label_count)
            return noisy_image, noisy_label
        elif error_type == "invalid label":
            noisy_image, noisy_label = generate_invalid_label_name_error(img, label_json, total_label_count)
            return None, noisy_label
    else:
        # generate shape error with a certain probability
        labels = label_json.get("Labels", [])
        for label in labels:
            shapes = label.get("Shapes", [])
            for shape in shapes:
                if np.random.rand() < SHAPE_ERROR_PROB:
                    error_type = np.random.choice(["invalid shape type", "out of range points", "rotation", "scaling"])
                    if error_type == "invalid shape type":
                        generate_invalid_shape_type_error(img, shape, total_label_count)
                    elif error_type == "out of range points":
                        generate_out_of_range_points_error(img, shape, total_label_count)
                    elif error_type == "rotation":
                        generate_annotation_by_rotation(img, shape, total_label_count)
                    elif error_type == "scaling":
                        generate_annotation_by_scaling(img, shape, total_label_count)
                else:
                    total_label_count["total"] += 1

        return None, label_json