import os
import cv2
import numpy as np
import json
from src.DataCleanModel.feature_analysis.feature_validator import FeatureValidator


# fix random seed for reproducibility
np.random.seed(42)

DATA_FOLDER_LIST = ["DetR-large-scale.coco-segmentation", "DOS-semicon.coco-segmentation", \
                    "fariq-generalize-dos.coco-segmentation", \
                        "UNISEM-ZN1461.coco-segmentation"]

DATA_FOLDER = "test_cases/RoboFlowSyntheticData"
INSPECTION_LOG_CASE_LIST_PATH = "config/test_log_cases.txt"
TRAIN_LOG_CASE_LIST_PATH = "config/train_log_cases.txt"


def performance_test(log_caste_list_path:str):
    config_path = "config/data_validation_config.yaml"
    feature_validator = FeatureValidator(config_path)

    with open(log_caste_list_path, "r") as f:
        inspection_log_cases = [line.strip() for line in f.readlines()]

    json_result_summary = {}

    for log_case_folder in inspection_log_cases:
        result_info = {}
        print(f"Running feature validation for log case: {log_case_folder}")
        result_path = os.path.join(DATA_FOLDER, log_case_folder, "FeatureValidationResult")

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        else:
            # clear result path
            for file in os.listdir(result_path):
                file_path = os.path.join(result_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        # best strategy
        img_path = os.path.join(DATA_FOLDER, log_case_folder, "image.jpg")
        img = cv2.imread(img_path)
        label_path = os.path.join(DATA_FOLDER, log_case_folder, "label.json")
        with open(label_path, "r") as f:
            label_json = json.load(f)
        result_info = feature_validator(log_case_folder, img, label_json, result_folder=result_path)
        json_result_summary[log_case_folder.replace("/", "_")] = {
            "precision": result_info.get("Precision", 0) if result_info else 0,
            "recall": result_info.get("Recall", 0) if result_info else 0,
            "tp": result_info.get("True Positives Num", 0) if result_info else 0,
            "tn": result_info.get("True Negatives Num", 0) if result_info else 0,
            "fp": result_info.get("False Positives Num", 0) if result_info else 0,
            "fn": result_info.get("False Negatives Num", 0) if result_info else 0,
            "fp list": result_info.get("False Positives", []) if result_info else [],
            "fn list": result_info.get("False Negatives", []) if result_info else [],
        }
 
    # calculate average precision and recall for best strategy and random strategy
    # exclude 0 values in precsion and recall calculation since they indicate the case where no valid label is detected, which can be misleading for performance evaluation
    valid_precisions = [result["precision"] for result in json_result_summary.values() if result["precision"] > 0]
    valid_recalls = [result["recall"] for result in json_result_summary.values() if result["recall"] > 0]
    avg_precision_best = np.mean(valid_precisions) if valid_precisions else 0
    avg_recall_best = np.mean(valid_recalls) if valid_recalls else 0
    print(f"Average precision: {avg_precision_best}")
    print(f"Average recall: {avg_recall_best}")

    

if __name__ == "__main__":
    print("Running feature validation performance test for training log cases...")
    performance_test(TRAIN_LOG_CASE_LIST_PATH)
    print("Running feature validation performance test for inspection log cases...")
    performance_test(INSPECTION_LOG_CASE_LIST_PATH)