import os
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple
from src.data_validation_with_hybrid import DataValidationWithHybridStrategy


# fix random seed for reproducibility
np.random.seed(42)

DATA_FOLDER_LIST = ["DetR-large-scale.coco-segmentation", "DOS-semicon.coco-segmentation", \
                    "fariq-generalize-dos.coco-segmentation", \
                        "UNISEM-ZN1461.coco-segmentation"]

EXEMPLAR_FOLDER = "test_cases/RoboFlowConverted"
EXEMPLAR_RECOMMENDAR_LOG_CASE_FOLDER = "test_cases/RoboFlowConverted"
DATA_FOLDER = "test_cases/RoboFlowSyntheticData"
INSPECTION_LOG_CASE_LIST_PATH = "config/test_log_cases.txt"
TRAIN_LOG_CASE_LIST_PATH = "config/train_log_cases.txt"


# split data into kg list and inspection list
def split_data(log_case_folder:str, split_ratio:float=0.5):
    log_case_list = []
    for folder in DATA_FOLDER_LIST:
        for dir in os.listdir(os.path.join(log_case_folder, folder)):
            log_case_list.append(os.path.join(folder, dir))
    np.random.shuffle(log_case_list)
    split_index = int(len(log_case_list) * split_ratio)
    kg_log_cases = log_case_list[:split_index]
    inspection_log_cases = log_case_list[split_index:]
    return kg_log_cases, inspection_log_cases


def performance_test(recommendar_log_cases_list_path:str, 
                     inspection_log_cases_list_path:str, 
                     use_json_schema_validation:bool=True, 
                     use_feature_validation:bool=True, 
                     use_sam3_validation:bool=True,):
    config_path = "config/data_validation_config.yaml"
    
    inspection_log_cases = []
    with open(inspection_log_cases_list_path, "r") as f:
        log_cases = [line.strip() for line in f.readlines()]
        for log_case in log_cases:
            inspection_log_cases.append(log_case)

    hybrid_validator = DataValidationWithHybridStrategy(config_path,
                                                         EXEMPLAR_FOLDER,
                                                         EXEMPLAR_RECOMMENDAR_LOG_CASE_FOLDER,
                                                         recommendar_log_cases_list_path)


    inspection_folder_list = []
    for log_case in inspection_log_cases:
        inspection_folder_list.append((os.path.join(DATA_FOLDER, log_case), log_case))

    hybrid_validator_result_summary = {}

    for log_case_folder, log_case_name in inspection_folder_list:
        print(f"Running hybrid validation for log case: {log_case_name}")
        log_path = os.path.join(log_case_folder, "HybridDataValidationLog")
        result_path = os.path.join(log_case_folder, "HybridDataValidationResult")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            # clear log path
            for file in os.listdir(log_path):
                file_path = os.path.join(log_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        else:
            # clear result path
            for file in os.listdir(result_path):
                file_path = os.path.join(result_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        img_path = os.path.join(log_case_folder, "image.jpg")
        img = cv2.imread(img_path)
        lable_path = os.path.join(log_case_folder, "label.json")
        with open(lable_path, "r") as f:
            json_data = json.load(f)

        result_info = hybrid_validator(validation_case_folder=log_case_folder, 
                                       img=img,
                                       json_data=json_data,
                                       log_path=log_path, 
                                       result_path=result_path, 
                                       use_json_validator=use_json_schema_validation, 
                                       use_feature_validator=use_feature_validation,
                                       use_sam3_validator=use_sam3_validation)
        hybrid_validator_result_summary[log_case_name] = {
            "precision": result_info.get("Precision", 0) if result_info else 0,
            "recall": result_info.get("Recall", 0) if result_info else 0,
            "tp": result_info.get("True Positives Num", 0) if result_info else 0,
            "tn": result_info.get("True Negatives Num", 0) if result_info else 0,
            "fp": result_info.get("False Positives Num", 0) if result_info else 0,
            "fn": result_info.get("False Negatives Num", 0) if result_info else 0,
            "fp list": result_info.get("False Positives", []) if result_info else [],
            "fn list": result_info.get("False Negatives", []) if result_info else [],
        }
 
        with open("results/hybrid_validator_result_summary.json", "w") as f:
            json.dump(hybrid_validator_result_summary, f, indent=4)

    # calculate average precision and recall for best strategy and random strategy
    avg_precision_best = np.mean([result["precision"] for result in hybrid_validator_result_summary.values()])
    avg_recall_best = np.mean([result["recall"] for result in hybrid_validator_result_summary.values()])

    print(f"Average precision for hybrid validator: {avg_precision_best}")
    print(f"Average recall for hybrid validator: {avg_recall_best}")


if __name__ == "__main__":
    performance_test(TRAIN_LOG_CASE_LIST_PATH, INSPECTION_LOG_CASE_LIST_PATH, use_json_schema_validation=True, \
                     use_feature_validation=True, use_sam3_validation=True)