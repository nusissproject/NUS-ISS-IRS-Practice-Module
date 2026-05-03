# Generate Final Synthetic data with errors
import os
import json
import cv2
import numpy as np
from typing import Dict, Tuple
from copy import deepcopy
import shutil
from src.DataCleanModel.data_process.generate_synthetic_data_with_error import generate_synthetic_data_with_error

DATA_FOLDER_LIST = ["DetR-large-scale.coco-segmentation", "DOS-semicon.coco-segmentation", \
                    "fariq-generalize-dos.coco-segmentation", \
                        "UNISEM-ZN1461.coco-segmentation"]
UNIQUE_SAMPLE_NAME = "src/DataCleanModel/data_process/unique_word_2000.txt"
SRC_FOLDER = "test_cases/RoboFlowConverted"
TARGET_FOLDER = "test_cases/RoboFlowSyntheticData"



if __name__ == "__main__":
    folder_name_list = []
    total_label_count = {"total":0,"invalid label": 0, "blur": 0, "low_contrast": 0, "darken": 0, "invalid shape type": 0, "out of range points": 0, "rotation": 0, "scaling": 0}
    for folder in DATA_FOLDER_LIST:
        print(f"Processing folder: {folder}")
        src_dir = f"test_cases/RoboFlowConverted/{folder}"
        target_dir = f"{TARGET_FOLDER}/{folder}"

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for subfolder in os.listdir(src_dir):
            subfolder_path = os.path.join(src_dir, subfolder)
            target_subfolder_path = os.path.join(target_dir, subfolder)
            if not os.path.exists(target_subfolder_path):
                os.makedirs(target_subfolder_path)
            img_path = os.path.join(subfolder_path, "image.jpg")
            label_path = os.path.join(subfolder_path, "label_clean.json")
            if os.path.exists(img_path) and os.path.exists(label_path):
                img = cv2.imread(img_path)
                with open(label_path, "r") as f:
                    label_json = json.load(f)
                synthetic_img, synthetic_label_json = generate_synthetic_data_with_error(deepcopy(img), deepcopy(label_json), total_label_count)

                if synthetic_img is None:
                    # copy from src to target if no image error is generated
                    shutil.copy(img_path, os.path.join(target_subfolder_path, "image.jpg"))
                else:
                    cv2.imwrite(os.path.join(target_subfolder_path, "image.jpg"), synthetic_img)
                with open(os.path.join(target_subfolder_path, "label.json"), "w") as f:
                    json.dump(synthetic_label_json, f, indent=4)
        
    # save label count
    with open(os.path.join(TARGET_FOLDER, "label_count.json"), "w") as f:
        json.dump(total_label_count, f, indent=4)
    print("Completed generating synthetic data with errors.")
    print(f"Total labels: {total_label_count}")