import math
import numpy as np
from typing import Dict, Tuple, List
import json
import os

class ExemplarEntity:
    def __init__(self, name:str, image_embedding:np.ndarray, img_path:str, exemplar_roi:Tuple[int, int, int, int], wire_width:float):
        self.name = name
        self.image_embedding = image_embedding # For similarity search
        self.img_path = img_path
        self.exemplar_roi = exemplar_roi
        self.wire_width = wire_width
        self.attributes = {}

    def __repr__(self):
        return f"ExemplarEntity({self.name})"
    

class LogCaseEntity:
    def __init__(self, name:str, image_embedding:np.ndarray, img_path:str):
        self.name = name
        self.image_embedding = image_embedding # For similarity search
        self.img_path = img_path

    def __repr__(self):
        return f"LogCaseEntity({self.name})"
    

class ReferenceRelationship:
    def __init__(self, exemplar_name, log_case_name):
        self.exemplar_name = exemplar_name
        self.log_case_name = log_case_name

    def __repr__(self):
        return f"ReferenceRelationship({self.exemplar_name} -> {self.log_case_name})"


class CollaborationFilteringRecommender:
    def __init__(self):
        self.exemplar_entities = {}  # {id: Entity object}
        self.log_case_entities = {}  # {id: Entity object}
        self.edges = []     # List of Relationship objects

    def add_exemplar_entity(self, name:str, image_embedding:np.ndarray, img_path:str, exemplar_roi:Tuple[int, int, int, int], wire_width:float)->ExemplarEntity:
        entity = ExemplarEntity(name, image_embedding, img_path, exemplar_roi, wire_width)
        self.exemplar_entities[name] = entity
        return entity
    
    def add_log_case_entity(self, name:str, image_embedding:np.ndarray, img_path:str)->LogCaseEntity:
        entity = LogCaseEntity(name, image_embedding, img_path)
        self.log_case_entities[name] = entity
        return entity

    def add_reference_relationship(self, exemplar_name, log_case_name):
        if exemplar_name not in self.exemplar_entities or log_case_name not in self.log_case_entities:
            raise ValueError("Both entities must exist in the graph.")
       
        rel = ReferenceRelationship(exemplar_name, log_case_name)
        self.edges.append(rel)

    # --- Similarity Search ---
    def find_similar(self, log_case_query_embeddings, top_k=3):
        """Calculates Cosine Similarity between query and all entities."""
        scores = []
       
        for name, entity in self.log_case_entities.items():          
            # Simple Cosine Similarity
            a = np.array(log_case_query_embeddings)
            b = np.array(entity.image_embedding)
            similarity = np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            scores.append((name, similarity))
       
        # Sort by highest similarity
        similar_log_case_entity = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

        for name, score in similar_log_case_entity:
            # search relations with log case name
            related_exemplars = [rel.exemplar_name for rel in self.edges if rel.log_case_name == name]
        return related_exemplars
    
    def find_random(self):
        """Finds a random exemplar entity."""
        if not self.exemplar_entities:
            return None
        random_entity = np.random.choice(list(self.exemplar_entities.values()))
        return random_entity
    

    def build_exemplars_db(self, exemplar_folder_list:List[Tuple[str,str]], log_case_folder_list:List[Tuple[str,str]]):
        """Builds the knowledge graph from exemplar and log case data."""
        for exemplar_folder, exemplar_name in exemplar_folder_list:
            # load json data
            exemplar_img_path = os.path.join(exemplar_folder, "image.jpg")
            with open(os.path.join(exemplar_folder, "exemplar_info.json"), "r") as f:
                exemplar_info = json.load(f)
                self.add_exemplar_entity(
                    name=exemplar_name,
                    image_embedding=exemplar_info.get("image_embedding", []),
                    img_path=exemplar_img_path,
                    exemplar_roi=exemplar_info["roi"],
                    wire_width=exemplar_info.get("wire width", 0.0)
            )
        
        for log_case_folder, log_case_name in log_case_folder_list:
            log_case_img_path = os.path.join(log_case_folder, "image.jpg")
            with open(os.path.join(log_case_folder, "label_clean.json"), "r") as f:
                log_case_info = json.load(f)
                self.add_log_case_entity(
                    name=log_case_info["globalAttributes"].get("exemplar_name", []),
                    image_embedding=log_case_info["globalAttributes"].get("image_embedding", []),
                    img_path=log_case_img_path
                )
                self.add_reference_relationship(log_case_info["globalAttributes"].get("exemplar_name", []), log_case_name.replace("/", "_"))


