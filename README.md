# intrusion-detection-feature-selection-ml
Intrusion Detection using GA, ReliefF and Stacking Ensemble
# Intrusion Detection using Feature Selection and Stacking Ensemble Learning

Machine Learning based Intrusion Detection System using Genetic Algorithm and ReliefF feature selection with Stacking Ensemble Learning on the CIC IoMT 2024 dataset. The project focuses on improving intrusion detection performance using feature selection and ensemble methods.

---

## Project Overview

This project focuses on building an efficient Intrusion Detection System (IDS) using Machine Learning techniques. The main objective of this project is to improve classification performance on network intrusion data by combining feature selection techniques with ensemble learning.

The project uses Genetic Algorithm and ReliefF for feature selection and a stacking ensemble model to improve classification accuracy and overall model performance.

---

## Dataset

This project uses the **CIC IoMT 2024 dataset**, which contains network traffic data for intrusion detection in Internet of Medical Things (IoMT) environments.

The dataset includes the following traffic classes:

* Benign Traffic
* MQTT DoS
* MQTT DDoS
* Reconnaissance

Due to the large size of the dataset, a sampled dataset is included in this repository for demonstration and reproducibility.

---

## Data Preprocessing

The following preprocessing steps were performed on the dataset:

* Checked for missing values (no missing values were present)
* Dropped the **drate** column because it contained only zero values and had no variance
* Applied **Label Encoding** to convert categorical features into numerical format
* Applied **Feature Scaling** where required for machine learning models
* Used **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance

---

## Methodology

### Feature Selection

Two feature selection techniques were used:

* Genetic Algorithm (GA)
* ReliefF

These methods were used to select the most relevant features and reduce dimensionality.

---

### Machine Learning Models

#### Base Models

* Decision Tree
* XGBoost
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Naive Bayes

#### Ensemble Model

* Stacking Ensemble Classifier
* Meta-classifier: Decision Tree

The stacking model combines predictions from multiple base models to improve overall performance.

---

### Model Evaluation

The models were evaluated using:

* Stratified K-Fold Cross Validation
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Results

The performance of multiple machine learning models was compared using feature-selected datasets.

The stacking ensemble model achieved the best performance compared to individual models such as Decision Tree and XGBoost.

Key observations:

* Feature selection reduced unnecessary features and improved model efficiency
* SMOTE improved performance for minority classes
* Stacking ensemble improved overall classification performance
* The combination of Genetic Algorithm feature selection and stacking produced the best results

Model comparison tables and confusion matrices are available in the **results** folder.

---

## Project Structure

The repository is organized as follows:

* **01_notebooks/** – Jupyter notebooks for feature selection, stacking models, and experiments
* **02_data/** – Sample dataset used for demonstration
* **03_results/** – Model comparison tables and confusion matrices
* **04_src/** – Python scripts for preprocessing and model implementation
* **README.md** – Project documentation
* **requirements.txt** – Python dependencies required to run the project

---

## Requirements

Install the required libraries using:

```
pip install -r requirements.txt
```

Main libraries used:

* numpy
* pandas
* scikit-learn
* imbalanced-learn
* xgboost
* matplotlib

---

## Conclusion

This project demonstrates that combining feature selection techniques such as Genetic Algorithm and ReliefF with stacking ensemble learning significantly improves intrusion detection performance.

The approach helps reduce feature dimensionality, handle class imbalance, and improve classification accuracy, making it suitable for network intrusion detection systems.

---

## Author

**Anshika Chandrawanshi**
B.Tech Project – Intrusion Detection using Machine Learning
