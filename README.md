# Stroke Prediction using SVM, Bagging, and AdaBoost

![Python](https://img.shields.io/badge/Python-3.x-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)


## Highlights
- Comparison of three machine learning algorithms (SVM, Bagging, and AdaBoost)
- SMOTE is applied only on the training set to prevent data leakage
- Models are evaluated using F1 score, ROC AUC, and cross validation
- Model behavior is interpreted through feature importance analysis

## Overview
Stroke is one of the leading causes of death worldwide. Early identification of stroke risk factors can support better prevention strategies. This project aims to analyze and predict stroke using machine learning — SVM, Bagging, and AdaBoost — based
on patient medical data. Beyond comparing accuracy, this project focuses on identifying the most influential stroke risk factors and evaluating models using metrics that are suitable for imbalanced datasets.

## Dataset
The dataset includes demographic and clinical features for stroke prediction.

**Source:** [Stroke Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

| Item | Value |
|---|---|
| Samples | 5,110 |
| Features | 11 |
| Prediction Task | Binary Classification |
| Target | Stroke (0 = No Stroke, 1 = Stroke) |
| Class Imbalance | ~95% Non-Stroke vs ~5% Stroke |
| Missing Values | BMI (201 records, 3.9%) |

## Project Workflow
The project workflow is shown in the diagram below:

![Workflow](images/workflow.png)

## Data Preprocessing
The following preprocessing steps were applied:

- Remove the ID column
- Fill missing BMI values using mean imputation (assumed MCAR — only 3.9% of data is missing)
- Remove rare gender category ('Other', only 1 row)
- Apply One-Hot Encoding for categorical variables
- Split data into training and test sets
- Handle class imbalance:
    - Use stratify=y during train-test split
    - Apply SMOTE only on the training set to prevent data leakage

## Models
The following machine learning algorithms were used:

| Model	| Reason for Selection |
|---|---|
| SVM (linear kernel)	| Used as a baseline linear classifier to evaluate whether the dataset can be linearly separated |
| Bagging (tree-based ensemble)	| Chosen as a tree-based ensemble method that can capture non-linear patterns while reducing variance for a more stable model |
| AdaBoost (boosting ensemble)	| Chosen to evaluate whether sequential learning can improve the model's ability to handle difficult-to-classify samples |

## Evaluation Metrics
Due to high class imbalance, evaluation focuses on:

- F1 score
- ROC AUC
- Confusion matrix

> Accuracy is still reported as additional information,
> but is not used as the main metric due to class imbalance.

## Results

| Model |	Accuracy |	F1	| CV F1	| AUC	| CV AUC |
|---|---|---|---|---|---|
| SVM |	0.95	| 0.00 ± 0.00	| 0.00 	| 0.43	| 0.57 ± 0.05 |
| Bagging	| 0.94	| 0.16 ± 0.04	| 0.09	| 0.81	| 0.79 ± 0.03 |
| AdaBoost	| 0.94	| 0.10 ± 0.05	| 0.13	| 0.84	| 0.81 ± 0.01 |

Although SVM achieved high accuracy, it was unable to identify stroke cases — F1 score of 0.00 and ROC AUC below 0.5. In contrast, tree-based ensemble methods showed better ability to distinguish stroke from non-stroke cases. AdaBoost achieved the highest ROC AUC in cross-validation, indicating that tree-based ensembles are better suited to capture non-linear patterns in this dataset compared to linear SVM.

## Feature Importance
Both Bagging and AdaBoost consistently identified **age** as the most important feature, followed by **average glucose level** and **BMI** — indicating that these variables contribute the most to the model's predictions.

**Top 3 Most Important Features:**
1. Age
2. Average Glucose Level
3. BMI

![Feature Importance Comparison](images/feature_importance_comparison.png)

## Key Findings
- The dataset has a high class imbalance (~95% vs ~5%)
- **Age** is the most influential feature for stroke prediction
- `ever_married_Yes` shows correlation with stroke (0.108), but this is likely explained by **age as a confounding variable** — individuals who are married tend to be older
- **High accuracy (95%) does not mean a good model** for imbalanced data — SVM's ROC AUC is only 0.42 (below random classifier)
- Tree-based ensemble methods significantly outperform linear SVM on this dataset
- ROC AUC shows that ensemble models can distinguish positive and negative cases reasonably well, but with the standard threshold of 0.5, results are not yet optimal — as reflected in the low F1 score
- **AdaBoost is the most consistent model** (CV AUC: 0.81 ± 0.01)

## Lessons Learned
- Algorithm selection should be based on data characteristics and the problem at hand, not personal preference
- The importance of preventing data leakage
- How to apply SMOTE correctly
- The difference between accuracy, F1 score, and ROC AUC
- Model validation using cross-validation
- The challenges of working with imbalanced health datasets

## Limitations
- Model performance may be affected by the relatively small dataset
- F1 score for the stroke class is still low due to extreme class imbalance (~5% positive cases)
- Hyperparameter tuning was not performed
- Full preprocessing pipeline was not implemented

## Future Work
- Hyperparameter tuning
- Implement a complete preprocessing pipeline
- Threshold optimization
- Precision-Recall curve analysis
- SHAP analysis
- Evaluate additional ensemble models such as Balanced Random Forest, XGBoost, or LightGBM
- Use a larger dataset

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn
