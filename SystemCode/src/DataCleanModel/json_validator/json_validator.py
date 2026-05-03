import json

import numpy as np
from shapely import box
import yaml
import os
import cv2
from jsonschema import validate, ValidationError
from typing import Dict, List, Tuple, Any
from src.DataCleanModel.utils.metrics import compute_json_validator_precision_recall_with_gt
from src.DataCleanModel.utils.ai_labeler_json_processer import read_json_data
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo


LABEL_SCHEMA = {
                "type":"object",
                "required":[
                    "Label",
                    "Shapes"
                ],
                "properties":{
                    "Label":{
                        "type":"string",
                        "enum":[
                            "wire"
                        ]
                    },
                    "Shapes":{
                        "type":"array",
                         "minItems":1
                         },
                    "additionalProperties":True
            }
        }


SHAPE_SCHEMA = {
                "type":"object",
                "required":[
                "Type",
                "points",
                "Attributes"],
                "properties":{
                    "Type":{
                        "type":"string",
                        "enum":[
                            "polygon"
                            ]
                        },
                    "proposed points":{
                        "type":"array",
                        "minItems":1,
                        "items":{
                            "type":"array",
                            "minItems":2,
                            "maxItems":2,
                            "items":{
                                "type":"number",
                                "minimum":0,
                                "maximum":5120
                                }
                            }
                        },
                    "points":{
                        "type":"array",
                        "minItems":1,
                        "items":{
                            "type":"array",
                            "minItems":2,
                            "maxItems":2,
                            "items":{
                                "type":"number",
                                "minimum":0,
                                "maximum":5120
                                }
                            }
                        },
                    "Attributes":{
                        "type":"object",
                        "required":[
                            "category_id"
                            ],
                        "properties":{
                            "category_id":{
                                "type":"number"
                                }
                            }
                        },
                        "additionalProperties":True
                    }
                }

class DataValidationWithJsonSchema:
    def __init__(self, config_source: str | dict[str, Any]):
        if isinstance(config_source, str):
            if not os.path.isfile(config_source):
                raise FileNotFoundError(f"Config file not found: {config_source}")
            with open(config_source) as f:
                data = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            data = config_source
        else:
            raise ValueError("Invalid config source type. Must be a file path or a dictionary.")

        self.labels = data["labels"] 

    def __call__(self, 
                 inspection_path:str,
                 img:np.ndarray, 
                 json_data:Dict, 
                 result_folder:str)->Dict[int, Tuple[str, List[Tuple[float, float, float, float]]]]:
        result_info = {}
        result_info["inspection_path"] = inspection_path
        result_info["Labels with Issue"] = []
        labels_evaluated = self.labels + ["InvalidLabelName"]
        json_array, _, _ = read_json_data(json_data, labels_evaluated)
        Labels = json_data.get("Labels", [])
        for label in Labels:
            try:
                validate(instance=label, schema=LABEL_SCHEMA)
            except ValidationError as e:
                for shape in label.get("Shapes", []):
                    label_with_issue = {}
                    label_with_issue["id"] = shape.get("Attributes", {}).get("category_id", "NA")
                    s = ShapeInfo(shape.get("Type", "NA"))
                    s.set_general_shape_point(shape.get("points", []), "wire")
                    debug_image = img.copy()
                    s.draw_shape(debug_image, color=(0,0,255), skip_type_error=True)
                    points = np.array(s.m_points)
                    crop_ul = (max(int(min(points[:,0])) - 10, 0), max(int(min(points[:,1])) - 10, 0))
                    crop_lr = (min(int(max(points[:,0])) + 10, debug_image.shape[1]), min(int(max(points[:,1])) + 10, debug_image.shape[0]))
                    debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
                    cv2.imwrite(os.path.join(result_folder, f"label_issue_{id}.jpg"), debug_image[:,:,::-1])
                    label_with_issue["issue_image"] = os.path.join(result_folder, f"label_issue_{id}.jpg")
                    label_with_issue['data failure reason'] = f"{e.message}"
                    result_info["Labels with Issue"].append(label_with_issue)
                
        for label in Labels:
            shapes = label.get("Shapes", [])
            for shape in shapes:
                try:
                    validate(instance=shape, schema=SHAPE_SCHEMA)
                except ValidationError as e:
                    label_with_issue = {}
                    label_with_issue["id"] = shape.get("Attributes", {}).get("category_id", "NA")
                    s = ShapeInfo(shape.get("Type", "NA"))
                    s.set_general_shape_point(shape.get("points", []), "wire")
                    debug_image = img.copy()
                    s.draw_shape(debug_image, color=(0,0,255), skip_type_error=True)
                    points = np.array(s.m_points)
                    crop_ul = (max(int(min(points[:,0])) - 10, 0), max(int(min(points[:,1])) - 10, 0))
                    crop_lr = (min(int(max(points[:,0])) + 10, debug_image.shape[1]), min(int(max(points[:,1])) + 10, debug_image.shape[0]))
                    debug_image = debug_image[crop_ul[1]:crop_lr[1], crop_ul[0]:crop_lr[0], :]
                    cv2.imwrite(os.path.join(result_folder, f"label_issue_{shape.get("Attributes", {}).get("category_id", "NA")}.jpg"), debug_image[:,:,::-1])
                    label_with_issue["issue_image"] = os.path.join(result_folder, f"label_issue_{shape.get("Attributes", {}).get("category_id", "NA")}.jpg")
                    label_with_issue['data failure reason'] = f"{e.message}"
                    result_info["Labels with Issue"].append(label_with_issue)
        
        tp,tn,fp,fn,precision, recall = compute_json_validator_precision_recall_with_gt(json_array, result_info["Labels with Issue"], result_info, img, result_folder)

        result_info["True Positives Num"] = tp
        result_info["True Negatives Num"] = tn  
        result_info["False Positives Num"] = fp
        result_info["False Negatives Num"] = fn
        result_info["Precision"] = precision
        result_info["Recall"] = recall

        print(f"JSON Precision: {precision}, Recall: {recall}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        # save result info to json
        with open(os.path.join(result_folder, "json_data_validation_result.json"), "w") as f:
            json.dump(result_info, f, indent=4)

        return result_info
                  
