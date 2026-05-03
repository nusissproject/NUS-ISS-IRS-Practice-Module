# build exemplar for every log case

from email.mime import image

from click import Tuple

from src.DataCleanModel.sam3_validation.data_validation_with_sam import DataValidationWithFoundationModel
from typing import List, Tuple
import numpy as np
import cv2

def generate_exemplar_info(cv_model: DataValidationWithFoundationModel, img:np.ndarray, label_json:dict)->Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    exemplar_info = {}
    # extract whole image embeddings for knowledge graph search
    # resize image to 1008 x 1008 with center padding to maintain aspect ratio using opencv
    h, w, _ = img.shape
    scale = 1008 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    padded_img = np.zeros((1008, 1008, 3), dtype=np.uint8)
    padded_img[(1008 - new_h) // 2:(1008 - new_h) // 2 + new_h, (1008 - new_w) // 2:(1008 - new_w) // 2 + new_w, :] = resized_img
    exemplar_info["image_embedding"] = cv_model.get_image_embedding(padded_img).tolist()
    exemplar_roi = cv_model.select_least_overlapped_object_roi_with_json(label_json, label_name_list = ["wire"])
    exemplar_info["roi"] = [exemplar_roi]
    exemplar_info["wire width"] = label_json["globalAttributes"].get("wire width", None)
    return exemplar_info