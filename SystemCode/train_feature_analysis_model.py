import csv
import os
import cv2
import json
import numpy as np
from src.DataCleanModel.feature_analysis.feature_validator import FeatureValidator
import pickle

TRAIN_LOG_CASE_LIST = "config/recommendar_db_log_cases.txt"
TEST_LOG_CASE_LIST = "config/inspection_log_cases.txt"
TRAIN_LOG_CASE_FOLDER = "test_cases/RoboFlowSyntheticData"
TEST_LOG_CASE_FOLDER = "test_cases/RoboFlowSyntheticData" 

def feature_extraction(src_folder:str, log_case_list_path:str, csv_file_path:str):
    config_path = "config/data_validation_config.yaml"
    feature_validator = FeatureValidator(config_path)

    with open(log_case_list_path, "r") as f:
        log_cases = [line.strip() for line in f.readlines()]

    # run feature validator on train log cases and test log cases, and save the feature values to a json file for later analysis
    feature_list = []
    for log_case in log_cases:
        log_case_folder = os.path.join(src_folder, log_case)
        print(f"Running feature analysis for log case: {log_case}")
        img_path = os.path.join(log_case_folder, "image.jpg")
        img = cv2.imread(img_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        with open(os.path.join(log_case_folder, "label.json"), "r") as f:
            label_json = json.load(f)
        labels = label_json.get("Labels", [])
        for label in labels:
            for shape in label.get("Shapes", []):
                features = feature_validator.compute_shape_features(hsv_img, shape)
                features["log_case"] = log_case
                features["shape_id"] = shape.get("Attributes", {}).get("category_id", "NA")
                if shape.get("Attributes", {}).get("Label Error", None) is not None:
                    features["label"] = True
                else:                    
                    features["label"] = False
                feature_list.append(features)

    # save feature values to csv file
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["log_case", "shape_id", "stroke_width", "stroke_length", "average_hue", \
                       "hue_sd", "average_saturation", "saturation_sd", "average_value", "value_sd", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_list)
    print(f"Feature analysis completed. Feature values saved to {csv_file_path}")


def feature_analysis(csv_file_path:str, prefix:str=""):
    # draw histogram for every feature and use different color for label error and non-label error
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_file_path)
    feature_names = ["stroke_width", "stroke_length", "average_hue", "hue_sd", "average_saturation", "saturation_sd", "average_value", "value_sd"]
    for feature_name in feature_names:
        plt.figure(figsize=(10, 6))
        plt.hist(df[df["label"] == True][feature_name], bins=30, alpha=0.5, label="Label Error", color="red")
        plt.hist(df[df["label"] == False][feature_name], bins=30, alpha=0.5, label="No Label Error", color="blue")
        plt.xlabel(feature_name)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {feature_name}")
        plt.legend()
        plt.savefig(f"results/{prefix}{feature_name}_histogram.png")
        plt.close()
    print("Feature analysis completed. Histograms saved to results folder.")


def decision_tree_classifier(train_csv_path:str, test_csv_path:str, k:int):
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_score, recall_score

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    feature_names = ["stroke_width", "stroke_length", "average_value", "value_sd"]
    X_train = train_df[feature_names].values
    y_train = train_df["label"].values
    X_test = test_df[feature_names].values
    y_test = test_df["label"].values

    # map label True to 1 and False to 0
    y_train = np.where(y_train == True, 1, 0)
    y_test = np.where(y_test == True, 1, 0)

    dt = DecisionTreeClassifier(criterion="gini", class_weight={0:k, 1:1}, max_depth=5) # avoid false positive by assigning higher weight to negative class
    dt.fit(X_train, y_train)

    # calculate precision, recall, false positives and false negatives on train set
    y_train_pred = dt.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_fp = np.sum((y_train == 0) & (y_train_pred == 1))
    train_fn = np.sum((y_train == 1) & (y_train_pred == 0))
    print(f"Train: Precision: {train_precision}, Recall: {train_recall}, False Positives: {train_fp}, False Negatives: {train_fn}")

    y_pred = dt.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # calculate fp and fn
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    print(f"Test: Precision: {precision}, Recall: {recall}, False Positives: {fp}, False Negatives: {fn}")

    # visualize the decision tree
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_names, class_names=["No Label Error", "Label Error"], filled=True)
    plt.savefig("results/decision_tree.png")
    plt.close()
    print("Decision tree visualization saved to results/decision_tree.png")

    # save the trained model to a file
    with open("config/decision_tree_model.pkl", "wb") as f:
        pickle.dump(dt, f)
    print("Trained decision tree model saved to config/decision_tree_model.pkl")


def test_trained_classifier(model_path:str, csv_file_path:str):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score

    with open(model_path, "rb") as f:
        dt = pickle.load(f)

    test_df = pd.read_csv(csv_file_path)
    feature_names = ["stroke_width", "stroke_length", "average_value", "value_sd"]
    X_test = test_df[feature_names].values
    y_test = test_df["label"].values

    # map label True to 1 and False to 0
    y_test = np.where(y_test == True, 1, 0)

    y_pred = dt.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # calculate fp and fn
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    print(f"Test: Precision: {precision}, Recall: {recall}, False Positives: {fp}, False Negatives: {fn}")

if __name__ == "__main__":
    #feature_extraction(TRAIN_LOG_CASE_FOLDER,TRAIN_LOG_CASE_LIST, "results/train_feature_values.csv")
    #feature_extraction(".", TEST_LOG_CASE_LIST, "results/test_feature_values.csv")
    #feature_analysis("results/train_feature_values.csv", prefix="train_")
    #feature_analysis("results/test_feature_values.csv", prefix="test_")
    train_csv_path = "results/train_feature_values.csv"
    test_csv_path = "results/test_feature_values.csv"
    decision_tree_classifier(train_csv_path, test_csv_path, k=100)
    test_trained_classifier("config/decision_tree_model.pkl", test_csv_path)