# data validation with hybrid strategy, if either json validator or feature validator predicts a shape as label error, we consider it as label error in hybrid results, and send it for human review. This can be useful for understanding the upper bound performance if we consider any shape that is predicted as label error by either validator as label error and send it for human review, which can help us understand how many issues may require human review and how many of them can be correctly identified by at least one of the validators.

import os
import cv2
import numpy as np
import json
import yaml
from src.DataCleanModel.sam3_validation.data_validation_with_sam import DataValidationWithFoundationModel
from src.DataCleanModel.exemplar_recommender.collaboration_filtering import ExemplarEntity, LogCaseEntity,CollaborationFilteringRecommender
from src.data_validation_with_sam_and_exemplar_rec import DataValidationWithSamAndExemplarRec
from src.DataCleanModel.json_validator.json_validator import DataValidationWithJsonSchema
from src.DataCleanModel.feature_analysis.feature_validator import FeatureValidator
from src.DataCleanModel.utils.metrics import compute_hybrid_precision_recall_with_gt
from typing import Dict, List, Tuple
from src.DataCleanModel.utils.ai_labeler_json_processer import read_json_data

class DataValidationWithHybridStrategy:
    def __init__(self, config_path:str, 
                 exemplar_folder:str,
                 log_case_folder:str, 
                 recommendar_db_list_path:str):
        if isinstance(config_path, str):
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path) as f:
                data = yaml.safe_load(f)
        elif isinstance(config_path, dict):
            data = config_source
        else:
            raise ValueError("Invalid config source type. Must be a file path or a dictionary.")
        self.sam3_validator = DataValidationWithSamAndExemplarRec(config_path, exemplar_folder, log_case_folder, recommendar_db_list_path)
        self.json_validator = DataValidationWithJsonSchema(config_path)
        self.feature_validator = FeatureValidator(config_path)
        self.labels = data["labels"] 

    
    def merge_results(self, sam3_result_info:Dict, json_result_info:Dict, feature_result_info:Dict) -> Dict:
        merged_result_info = {}
        merged_result_info["Labels with Issue"] = []
        if sam3_result_info:
            sam3_label_with_issue = {x.get("id", "NA"): (x.get("data failure reason", "NA"), x.get("issue_image", "NA")) for x in sam3_result_info.get("Labels with Issue", [])}
        else:
            sam3_label_with_issue = {}
        if json_result_info:
            json_label_with_issue = {x.get("id", "NA"): (x.get("data failure reason", "NA"), x.get("issue_image", "NA")) for x in json_result_info.get("Labels with Issue", [])}
        else:
            json_label_with_issue = {}
        if feature_result_info:
            feature_label_with_issue = {x.get("id", "NA"): (x.get("data failure reason", "NA"), x.get("issue_image", "NA")) for x in feature_result_info.get("Labels with Issue", [])}
        else:
            feature_label_with_issue = {}
        all_label_ids = set(list(sam3_label_with_issue.keys()) + list(json_label_with_issue.keys()) + list(feature_label_with_issue.keys()))
        for label_id in all_label_ids:
            data_failure_reason = ""
            issue_image = None
            if label_id in sam3_label_with_issue:
                data_failure_reason += f"SAM3 Validator: {sam3_label_with_issue[label_id][0]}. "
                issue_image = sam3_label_with_issue[label_id][1]
            if label_id in json_label_with_issue:
                data_failure_reason += f"JSON Validator: {json_label_with_issue[label_id][0]}. "
                issue_image = json_label_with_issue[label_id][1]
            if label_id in feature_label_with_issue:
                data_failure_reason += f"Feature Validator: {feature_label_with_issue[label_id][0]}. "
                issue_image = feature_label_with_issue[label_id][1]
            merged_result_info["Labels with Issue"].append({
                "id": label_id,
                "data failure reason": data_failure_reason,
                "issue_image": issue_image
            })
        return merged_result_info

    
    def __call__(self,
                validation_case_folder:str,
                img:np.ndarray,
                json_data:Dict,
                kg_strategy:str="best",
                use_json_validator:bool=True,
                use_feature_validator:bool=True,
                use_sam3_validator:bool=True,
                log_path:str=None,
                result_path:str=None):
        labels_evaluated = self.labels + ["InvalidLabelName"]
        json_array, _, _ = read_json_data(json_data, labels_evaluated)

        result_info_list = []
        if use_sam3_validator:
            sam3_result_info = self.sam3_validator(validation_case_folder, img=img, kg_strategy=kg_strategy, log_path=log_path, result_path=result_path)
            if sam3_result_info:
                result_info_list.append(sam3_result_info)
        if use_json_validator:
            json_result_info = self.json_validator(validation_case_folder, img=img, json_data=json_data, result_folder=result_path)
            if json_result_info:
                result_info_list.append(json_result_info)
        if use_feature_validator:
            feature_result_info = self.feature_validator(validation_case_folder, img=img, json_data=json_data, result_folder=result_path)
            if feature_result_info:
                result_info_list.append(feature_result_info)
        merged_result_info = self.merge_results(sam3_result_info if use_sam3_validator else None, 
                                                json_result_info if use_json_validator else None, 
                                                feature_result_info if use_feature_validator else None)
        merged_result_info["inspection_path"] = validation_case_folder

        tp,tn,fp,fn,precision, recall = compute_hybrid_precision_recall_with_gt(json_array, result_info_list, merged_result_info, img, result_path)

        merged_result_info["True Positives Num"] = tp
        merged_result_info["True Negatives Num"] = tn  
        merged_result_info["False Positives Num"] = fp
        merged_result_info["False Negatives Num"] = fn
        merged_result_info["Precision"] = precision
        merged_result_info["Recall"] = recall

        print(f"Hybrid Precision: {precision}, Recall: {recall}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        # save result info to json
        with open(os.path.join(result_path, "hybrid_data_validation_result.json"), "w") as f:
            json.dump(merged_result_info, f, indent=4)

        return merged_result_info