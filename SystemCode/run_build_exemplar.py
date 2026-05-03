import os
import json
import cv2
import shutil

from src.DataCleanModel.exemplar_recommender.build_exemplar import generate_exemplar_info
from src.DataCleanModel.sam3_validation.data_validation_with_sam import DataValidationWithFoundationModel


DATA_FOLDER_LIST = ["DetR-large-scale.coco-segmentation", "DOS-semicon.coco-segmentation", \
                    "fariq-generalize-dos.coco-segmentation", \
                        "UNISEM-ZN1461.coco-segmentation"]

UNIQUE_SAMPLE_NAME = "src/DataCleanModel/data_process/unique_word_2000.txt"
SRC_FOLDER = "test_cases/RoboFlowConverted"
TARGET_FOLDER = "test_cases/RoboFlowConverted"

if __name__ == "__main__":
    config_path = "config/data_validation_config.yaml"
    cv_model = DataValidationWithFoundationModel(config_path)

    for folder in DATA_FOLDER_LIST:
        print(f"Processing folder: {folder}")
        src_dir = f"test_cases/RoboFlowConverted/{folder}"
        target_dir = f"{TARGET_FOLDER}/{folder}"

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for subfolder in os.listdir(src_dir):
            subfolder_path = os.path.join(src_dir, subfolder)
            print(subfolder_path)
            target_subfolder_path = os.path.join(target_dir, subfolder)
            if not os.path.exists(target_subfolder_path):
                os.makedirs(target_subfolder_path)
            img_path = os.path.join(subfolder_path, "image.jpg")
            label_path = os.path.join(subfolder_path, "label_clean.json")
            if os.path.exists(img_path) and os.path.exists(label_path):
                img = cv2.imread(img_path)
                with open(label_path, "r") as f:
                    label_json = json.load(f)
                exemplar_info = generate_exemplar_info(cv_model, img, label_json)
                with open(os.path.join(target_subfolder_path, "exemplar_info.json"), "w") as f:
                    json.dump(exemplar_info, f, indent=4)
            #shutil.copy(img_path, os.path.join(target_subfolder_path, "image.jpg"))
            # generate visualization of exemplar roi and save to target folder
            debug_image = img.copy()
            for roi in exemplar_info.get("roi", []):
                cv2.rectangle(debug_image, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0,255,0), 2)
            cv2.imwrite(os.path.join(target_subfolder_path, "exemplar_visualization.jpg"), debug_image)
            # update exemplar_name in log case
            label_json["globalAttributes"]["exemplar_name"] = f"{folder}_{subfolder}"
            label_json["globalAttributes"]["image_embedding"] = exemplar_info.get("image_embedding", [])
            label_json["globalAttributes"]["exemplar_roi"] = []
            with open(label_path, "w") as f:
                json.dump(label_json, f, indent=4)

        
    print("Completed generating exemplar info.")