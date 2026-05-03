
## SECTION 1 : PROJECT TITLE
## Data Quality Assurance System for Wire Bonding Inspection

<img src="SystemCode/clips/static/data_validator_gui.jpg"
     style="float: left; margin-right: 0px;" />

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
Inspecting wire bonding via Automated Optical Inspection (AOI) is a challenging task.  The large amount of image and annotation data are collected to train edge deep learning models for machine deployment and to validate machine software output through continuous integration. The data is subject to image and annotation quality issues resulting in sub-optimal model performance and diminished software reliability. To resolve these inefficiencies, this project implements a hybrid framework that integrates a rule-based expert system, decision tree classifier leveraging handcrafted features, and a finetune Segment Anything Model 3 (SAM3). To handle high-resolution image incompatible with SAM3 input size, the system utilizes a sophisticated partitioning and merging strategy optimized through hierarchical clustering, bin packing, and genetic algorithms. Furthermore, the SAM3’s image exemplar recommendation system is developed by collaborative filtering. Data preparation utilizes active learning for precise curation alongside the simulation of quality-compromised data to rigorously validate system performance.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Zheng Bo | A0339761U | SAM3 finetune inference pipeline GUI     | zhengbo@u.nus.edu |
| Sun Yan  | A0340238L | Json Schema validator decision tree video| sun.yan@u.nus.edu |


---

## SECTION 4 : USE CASE DEMO

Refer to Video\Demo Video.wmv

---

## SECTION 5 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`


## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`


