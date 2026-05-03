import os
import json
import cv2

from src.DataCleanModel.sam3_validation.data_validation_with_sam import DataValidationWithFoundationModel
from src.DataCleanModel.data_process.database_curation import data_curation

DATA_FOLDER_LIST = ["DetR-large-scale.coco-segmentation", "DOS-semicon.coco-segmentation", \
                    "fariq-generalize-dos.coco-segmentation", \
                        "UNISEM-ZN1461.coco-segmentation"]
UNIQUE_SAMPLE_NAME = "src/DataCleanModel/data_process/unique_word_2000.txt"
FOLDER_SAMPLE_NUMBER = [139, 41, 39, 96, 693, 50]
SRC_FOLDER = "test_cases/RoboFlowConverted"


if __name__ == "__main__":
    config_path = "config/data_validation_config.yaml"
    cv_model = DataValidationWithFoundationModel(config_path)
    folder_name_list = []

    for folder in DATA_FOLDER_LIST:
        for dir in os.listdir(os.path.join(SRC_FOLDER, folder)):
            data_folder = os.path.join(SRC_FOLDER, folder, dir)
            log_path = os.path.join(data_folder, "DataCurationLog")
            print(f"Processing folder: {data_folder}")
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            try:
                data_curation(data_folder, cv_model, log_path=log_path)
            except Exception as e:
                print(f"Error occurred while processing {data_folder}: {e}")

    print("Completed data curation.")