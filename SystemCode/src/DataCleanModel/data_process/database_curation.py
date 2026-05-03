# Curate dataset for database construction using SAM3 and manual verification

import json
import os
import cv2
from copy import deepcopy
from typing import Dict, Tuple
from PIL import Image
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo, LabelInfo, AnnotationFileGenerator, create_thickness_polygon
from src.DataCleanModel.utils.ai_labeler_json_processer import read_json_data, read_json_file
import numpy as np
from src.DataCleanModel.sam3_validation.data_validation_with_sam import DataValidationWithFoundationModel
from src.DataCleanModel.partition_merge.partition import Partition

def data_curation(data_folder, cv_model:DataValidationWithFoundationModel, log_path=None):
    # open image and load json annotation
    img = cv2.imread(os.path.join(data_folder, "image.jpg"))
    annotation = None
    with open(os.path.join(data_folder, "Label.json"), "r") as f:
        annotation = json.load(f)
    
    if img is None or annotation is None:
        print(f"Failed to load image or annotation from {data_folder}")
        return
    
    partitions, exemplar_rois, resize_img = cv_model.prepare_for_frame_with_bond_program(img, annotation, log_folder=log_path)

    if partitions is None or len(exemplar_rois) == 0:
        print(f"Failed to prepare for frame with bond program for {data_folder}")
        return
    
    if log_path is not None:
        debug_img = deepcopy(resize_img)
        for p in partitions:
            center = [int(p.center[0]), int(p.center[1])]
            cv2.circle(debug_img, center, 5, (0, 255, 0), -1)
            c1 = [max(0, int(p.center[0] - p.size[0]//2)), max(0, int(p.center[1] - p.size[1]//2))]
            c2 = [min(debug_img.shape[1], int(p.center[0] + p.size[0]//2)), min(debug_img.shape[0], int(p.center[1] + p.size[1]//2))]
            cv2.rectangle(debug_img, c1, c2, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(log_path, "partition_output.jpg"), debug_img)

    
    cv_model.data_curation_with_concept_prompt_bond_program(data_folder, img, annotation, partitions, img, exemplar_rois, log_folder=log_path)