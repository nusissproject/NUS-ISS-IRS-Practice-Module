from skimage.morphology import medial_axis
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import yaml
import os
import pickle
import json
from src.DataCleanModel.utils.ai_labeler_json_processer import read_json_data
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo
from src.DataCleanModel.utils.metrics import compute_feature_validator_precision_recall_with_gt


class FeatureValidator:
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
        self.decision_tree_model_path = data.get("decision_tree_model_path", None)
        if self.decision_tree_model_path is not None and os.path.isfile(self.decision_tree_model_path):
            with open(self.decision_tree_model_path, "rb") as f:
                self.classifier = pickle.load(f)
        self.labels = data["labels"]

    def compute_shape_features(self, hsv_img: np.ndarray, shape: Dict) -> Dict[str, float]:
        """
        shape: A dictionary containing the shape information, including the points and the type of the shape.
        Returns a dictionary containing the computed features, such as average stroke width.
        """
        features = {}
        points = shape.get("points", [])
        if not points:
            return {"average_stroke_width": None}
        
        height, width, _ = hsv_img.shape
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill the mask based on the shape type (e.g., polygon)
        # This is a simplified example; you may need to handle different shape types accordingly
        if shape.get("Type") == "polygon" or shape.get("Type") == "InvalidShapeType":
            poly_np = np.array(shape.get("points", [])).astype(np.int32)
            cv2.fillPoly(mask, [poly_np], 1)
        
        # Compute the medial axis and distance transform
        skel, distance = medial_axis(mask, return_distance=True)
        
        # Calculate average stroke width
        stroke_widths = distance[skel > 0]
        average_stroke_width = np.mean(stroke_widths) if len(stroke_widths) > 0 else None
        
        features["stroke_width"] = average_stroke_width

        # Calculate stroke lenght
        features["stroke_length"] = np.sum(skel)

        # Compute hue over mask area
        hue_values = hsv_img[:, :, 0][mask > 0]
        features["average_hue"] = np.mean(hue_values) if len(hue_values) > 0 else None

        # compute hue sd
        features["hue_sd"] = np.std(hue_values) if len(hue_values) > 0 else None

        # compute saturation over mask area
        saturation_values = hsv_img[:, :, 1][mask > 0]
        features["average_saturation"] = np.mean(saturation_values) if len(saturation_values) > 0 else None

        # compute saturation sd
        features["saturation_sd"] = np.std(saturation_values) if len(saturation_values) > 0 else None

        # compute value over mask area
        value_values = hsv_img[:, :, 2][mask > 0]
        features["average_value"] = np.mean(value_values) if len(value_values) > 0 else None
        
        features["value_sd"] = np.std(value_values) if len(value_values) > 0 else None

        return features
    
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
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feature_list = []
        label_list = []
        for label in Labels:
            for shape in label.get("Shapes", []):
                features = self.compute_shape_features(hsv_img, shape)
                features["shape_id"] = shape.get("Attributes", {}).get("category_id", "NA")
                if shape.get("Attributes", {}).get("Label Error", None) is not None:
                    features["label"] = 1
                else:                    
                    features["label"] = 0
                feature_list.append([features["stroke_width"], features["stroke_length"], features["average_value"], features["value_sd"]])
                label_list.append(features["label"])

        feature_np = np.array(feature_list)
        label_np = np.array(label_list)
        predictions = self.classifier.predict(feature_np)
        
        index = 0
        for label in Labels:
            for shape in label.get("Shapes", []):
                if predictions[index] == 1:
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
                    cv2.imwrite(os.path.join(result_folder, f"feature_issue_{shape.get('Attributes', {}).get('category_id', 'NA')}.jpg"), debug_image[:,:,::-1])
                    label_with_issue["issue_image"] = os.path.join(result_folder, f"feature_issue_{shape.get('Attributes', {}).get('category_id', 'NA')}.jpg")
                    label_with_issue['data failure reason'] = "Predicted as label error by feature validator"
                    result_info["Labels with Issue"].append(label_with_issue)
                index += 1

        tp,tn,fp,fn,precision, recall = compute_feature_validator_precision_recall_with_gt(json_array, result_info["Labels with Issue"], result_info, img, result_folder)
        result_info["True Positives Num"] = tp
        result_info["True Negatives Num"] = tn  
        result_info["False Positives Num"] = fp
        result_info["False Negatives Num"] = fn
        result_info["Precision"] = precision
        result_info["Recall"] = recall

        print(f"Feature Precision: {precision}, Recall: {recall}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        # save result info to json
        with open(os.path.join(result_folder, "feature_data_validation_result.json"), "w") as f:
            json.dump(result_info, f, indent=4)

        return result_info


    