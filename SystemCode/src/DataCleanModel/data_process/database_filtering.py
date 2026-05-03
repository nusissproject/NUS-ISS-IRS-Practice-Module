from typing import Dict, Tuple
import os
import json
from pycocotools.coco import COCO

import numpy as np
import cv2
from skimage.morphology import skeletonize
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo, LabelInfo, AnnotationFileGenerator
import shutil

CLOSE_POINT_TOLERANCE=10
LABEL_NAME_LIST = ["wire", "wires"]

def detect_end_points(binary_img: np.ndarray) -> np.ndarray:
    # 1  = Foreground (Hit)
    # -1 = Background (Miss)
    # 0  = Ignore (Don't care)
    kernels = [
        # Orthogonal
        np.array([[-1, -1, -1], [-1,  1, -1], [ 0,  1,  0]], dtype=np.int8), # North
        np.array([[ 0,  1,  0], [-1,  1, -1], [-1, -1, -1]], dtype=np.int8), # South
        np.array([[ 0, -1, -1], [ 1,  1, -1], [ 0, -1, -1]], dtype=np.int8), # East
        np.array([[-1, -1,  0], [-1,  1,  1], [-1, -1,  0]], dtype=np.int8), # West
        # Diagonal
        np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1,  1]], dtype=np.int8), # NE
        np.array([[-1, -1, -1], [-1,  1, -1], [ 1, -1, -1]], dtype=np.int8), # NW
        np.array([[ 1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.int8), # SE
        np.array([[-1, -1,  1], [-1,  1, -1], [-1, -1, -1]], dtype=np.int8)  # SW
    ]

    output_mask = np.zeros(binary_img.shape, dtype=np.uint8)

    for kernel in kernels:
        curr_hitmiss = cv2.morphologyEx(binary_img, cv2.MORPH_HITMISS, kernel)
        output_mask = cv2.bitwise_or(output_mask, curr_hitmiss)

    # extract the coordinates of endpoints
    endpoints = np.argwhere(output_mask > 0)
    endpoints_list = [[x,y] for y, x in endpoints]

    # merge endpoints that are close to each other (within 10 pixels) to avoid duplicates
    duplicate_endpoints = [0] * len(endpoints_list)
    for ep in endpoints_list:
        if duplicate_endpoints[endpoints_list.index(ep)]:
            continue
        for other_ep in endpoints_list:
            if ep == other_ep:
                continue
            if abs(ep[0] - other_ep[0]) <= CLOSE_POINT_TOLERANCE and abs(ep[1] - other_ep[1]) <= CLOSE_POINT_TOLERANCE:
                duplicate_endpoints[endpoints_list.index(other_ep)] = 1
    
    merged_endpoints = [[int(ep[0]), int(ep[1])] for idx, ep in enumerate(endpoints_list) if not duplicate_endpoints[idx]]

    return merged_endpoints


def compute_stroke_properties(polygons, width, height):
    """
    polygons: List of lists [[x1, y1, x2, y2, ...]]
    width, height: Dimensions of the image
    """
    # 1. Create a binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes = []
    for poly in polygons:
        poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [poly_np], 1)
        # find bounding box for each polygon and add to boxes list
        x, y, w, h = cv2.boundingRect(poly_np)
        boxes.append((x, y, x+w, y+h))
    
    if np.sum(mask) == 0:
        return 0, []

    # 2. Compute Distance Transform
    # Returns the distance from each pixel to the nearest 0-pixel (boundary)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 3. Get the Skeleton (Medial Axis)
    # This reduces the shape to a 1-pixel wide centerline
    skeleton = skeletonize(mask > 0)
    
    # 4. Extract distance values along the skeleton
    # Stroke width = 2 * distance to boundary
    stroke_widths = dist_map[skeleton] * 2

    endpoints = detect_end_points(skeleton.astype(np.uint8))
    
    # 5. Return the median of stroke widths > 0 and the endpoints
    return np.median(stroke_widths).tolist(), endpoints, boxes



def save_to_json(entry, target_dir):
    annotation_file = AnnotationFileGenerator()
    annotation_file.add_global_attribute("wire width", entry["median_stroke_width"])
    annotation_file.add_global_attribute("bond program", entry["bond_program"])
    label_info = LabelInfo('wire')
    for ann in entry['annotations']:
        for seg in ann['segmentation']:
            shape_info = ShapeInfo('polygon')
            shape_info.set_polygon_point(seg, image_list="image.jpg", attribubtes={"category_id": ann['category_id'], "area": ann['area'], "iscrowd": ann['iscrowd']})
            label_info.add_shape(shape_info)
    annotation_file.add_label_info(label_info)
    annotation_file.serialize_to_json(target_dir, "Label.json")


def load_coco_annotations(ann_file:str, img_dir:str, target_dir:str, folder_name_list, log_path:str=None):
    # Initialize COCO api for instance annotations
    coco = COCO(ann_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()

    idx = 0
    
    for img_id in img_ids:
        # reset entry for each image
        entry = {}
        # Load image metadata
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        full_path = os.path.join(img_dir, file_name)
        
        # Load annotation IDs for this specific image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        print(f"Processing Image ID: {img_id}, File: {file_name}, Annotations: {len(anns)}")
        # Structure the data
        entry = {
            'image_path': full_path,
            'image_id': img_id,
            'width': img_info['width'],
            'height': img_info['height'],
            'median_stroke_width': 0,
            "bond_program": [],
            'annotations': []
        }
        
        stroke_width_list = []
        endpoints_list = []
        for ann in anns:
            # COCO segmentation can be a list of polygons or RLE
            segmentation = ann['segmentation']
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            if category_name not in LABEL_NAME_LIST:
                continue
            
            seg_list = []
            for seg in segmentation:
                stroke_width, endpoints, boxes = compute_stroke_properties([seg], img_info['width'], img_info['height'])
                if stroke_width > 0:
                    stroke_width_list.append(stroke_width)
                if endpoints is not None and len(endpoints) == 2:
                    endpoints_list.append(endpoints)
                else: # use diagnonal of bounding box as fallback if endpoints cannot be detected
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        endpoints_list.append([[x1, y1], [x2, y2]])
                seg_list.append(np.array(seg).reshape(-1, 2).tolist())
            entry['annotations'].append({
                'category': category_name,
                'category_id': category_id,
                'segmentation': seg_list,
                'bbox': ann['bbox'], # [x, y, width, height]
                'area': ann['area'],
                'iscrowd': ann['iscrowd'],
            })
        entry["median_stroke_width"] = np.median(stroke_width_list).tolist() if stroke_width_list else 0
        entry["bond_program"] = endpoints_list

        if log_path:
            # plot end points on the image and save to log path for visualization
            img = cv2.imread(full_path)
            for ep_group in endpoints_list:
                for ep in ep_group:
                    cv2.circle(img, (ep[0], ep[1]), 3, (0, 0, 255), -1)
            log_file = os.path.join(log_path, f"{img_info['file_name']}_endpoints.png")
            cv2.imwrite(log_file, img)

        # save entry to ai labeler json format for potential use in training or visualization
        folder_name = folder_name_list[idx]
        if not os.path.exists(os.path.join(target_dir, folder_name)):
            os.makedirs(os.path.join(target_dir, folder_name))

        save_to_json(entry, os.path.join(target_dir, folder_name))
        # copy image to target dir as well
        target_image_path = os.path.join(target_dir, folder_name, "image.jpg")
        shutil.copy(full_path, target_image_path)

        # generate overlay image for visualization
        overlay = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)
        img = cv2.imread(full_path)
        for ann in entry['annotations']:
            for seg in ann['segmentation']:
                seg_np = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [seg_np], (0, 255, 0))
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        blended_path = os.path.join(target_dir, folder_name, "overlay.jpg")
        cv2.imwrite(blended_path, blended)
        idx += 1
        





