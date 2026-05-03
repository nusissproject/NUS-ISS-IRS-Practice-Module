from src.DataCleanModel.data_process.database_filtering import load_coco_annotations
import sys
import os

print(os.getcwd())

DATA_FOLDER_LIST = ["DetR-large-scale.coco-segmentation", "DOS-semicon.coco-segmentation", \
                    "fariq-generalize-dos.coco-segmentation",  \
                        "UNISEM-ZN1461.coco-segmentation"]
UNIQUE_SAMPLE_NAME = "src/DataCleanModel/data_process/unique_word_2000.txt"
FOLDER_SAMPLE_NUMBER = [139, 41, 39, 96, 693, 50]

if __name__ == "__main__":
    # Usage
    folder_name_list = []
    with open(UNIQUE_SAMPLE_NAME, "r") as f:
        folder_name_list = f.read().strip().split(",")
    # remove the white space around the folder names    
    folder_name_list = [name.strip() for name in folder_name_list]
    start_index = 0
    for folder, sample_number in zip(DATA_FOLDER_LIST, FOLDER_SAMPLE_NUMBER):
        print(f"Processing folder: {folder} with sample number: {sample_number}")
        ann_path = f"test_cases/RoboFlow/{folder}/train/_annotations.coco.json"
        img_path = f"test_cases/RoboFlow/{folder}/train"
        target_dir = f"test_cases/RoboFlowConverted/{folder}"
        word_start_idx = start_index
        word_end_idx = start_index + sample_number
        subfolder_name_list = folder_name_list[word_start_idx:word_end_idx]

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        coco_data = load_coco_annotations(ann_path, img_path, target_dir, subfolder_name_list)
        start_index += (sample_number+20)

    print("Completed loading and filtering coco annotations.")
