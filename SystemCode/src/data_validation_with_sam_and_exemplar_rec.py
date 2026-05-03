# run data validation with sam and exemplar recommender
from src.DataCleanModel.sam3_validation.data_validation_with_sam import DataValidationWithFoundationModel
from src.DataCleanModel.exemplar_recommender.collaboration_filtering import ExemplarEntity, LogCaseEntity,CollaborationFilteringRecommender
import os
import cv2
import numpy as np
import json


class DataValidationWithSamAndExemplarRec:
    def __init__(self, config_path:str, 
                 exemplar_folder:str,
                 log_case_folder:str, 
                 recommendar_db_list_path:str):
        self.cv_model = DataValidationWithFoundationModel(config_path)
        print("Foundation model loaded successfully.")
        self.exemplar_folder_list = []
        self.log_case_folder_list = []
        with open(recommendar_db_list_path, "r") as f:
            for line in f:
                exemplar_item = line.strip()
                self.exemplar_folder_list.append((os.path.join(exemplar_folder, exemplar_item), exemplar_item.replace("/", "_")))
                self.log_case_folder_list.append((os.path.join(log_case_folder, exemplar_item), exemplar_item))
        self.exemplar_recommendar = CollaborationFilteringRecommender()
        self.exemplar_recommendar.build_exemplars_db(self.exemplar_folder_list, self.log_case_folder_list)
        print("Knowledge graph built successfully.")
    
    def __call__(self,
                validation_case_folder:str,
                img:np.ndarray,
                kg_strategy:str="best",
                log_path:str=None,
                result_path:str=None):

        with open(os.path.join(validation_case_folder, "label.json"), "r") as f:
            label_json = json.load(f)
        h, w, _ = img.shape
        scale = 1008 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        padded_img = np.zeros((1008, 1008, 3), dtype=np.uint8)
        padded_img[(1008 - new_h) // 2:(1008 - new_h) // 2 + new_h, (1008 - new_w) // 2:(1008 - new_w) // 2 + new_w, :] = resized_img
        query_embedding = self.cv_model.get_image_embedding(padded_img).tolist()

        if kg_strategy == "best":
            recommendations = self.exemplar_recommendar.find_similar(query_embedding, top_k=1)
            exemplar_entity = self.exemplar_recommendar.exemplar_entities.get(recommendations[0], None)
        else: # random strategy
            exemplar_entity = self.exemplar_recommendar.find_random()

        print(f"{kg_strategy} exemplar:", exemplar_entity.name)
        exemplar_img_path = exemplar_entity.img_path
        exemplar_img = cv2.imread(exemplar_img_path)
        exemplar_roi = exemplar_entity.exemplar_roi 
        
        partitions, _, _ = self.cv_model.prepare_for_frame_with_bond_program(img, label_json, log_folder=log_path)
        return self.cv_model.data_validation_with_concept_prompt_bond_program(validation_case_folder, img, label_json, partitions, \
                                                                    exemplar_img, exemplar_roi, log_folder=log_path, result_folder=result_path)
        