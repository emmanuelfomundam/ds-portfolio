# 02 — Customer Churn Classifier

## Overview

A binary classification model that predicts whether a telecom customer will churn. Focuses on handling severe class imbalance via SMOTE and optimizing recall on the minority (churn) class through XGBoost with stratified cross-validated hyperparameter tuning.

**Best result: ROC-AUC 0.9344 (5-fold CV)**

## Dataset

**Telco Customer Churn**
- 7,043 customer records, 19 features
- Target: `Churn` (Yes / No) — 73.5% No / 26.5% Yes
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in `data/raw/`.

## Project Structure

```
02_customer_churn_classifier/
├── data/
│   ├── raw/                        # Raw CSV (not tracked by git)
│   └── processed/                  # Encoded/scaled data (not tracked by git)
├── notebooks/
│   └── churn_classifier.ipynb      # Full walkthrough notebook
├── src/
│   ├── preprocess.py               # load_and_clean(), encode_and_scale()
│   └── train.py                    # split_and_resample(), train_models(),
│                                   # evaluate_models(), tune_xgboost(),
│                                   # plot_feature_importance(), save_model()
├── models/                         # Serialized model (not tracked by git)
├── outputs/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Methodology

### Preprocessing
- Coerced `TotalCharges` to numeric; imputed 11 missing values with median
- Label encoded all categorical columns
- Scaled features with `StandardScaler`

### Class Imbalance
- Applied **SMOTE** to training set: 5,174 / 1,869 → 8,278 balanced samples

### Models Trained
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.8390 |
| Random Forest | ~0.91 |
| **XGBoost (tuned)** | **0.9344** |

### XGBoost Tuning
- 5-fold stratified `GridSearchCV`
- 16 combinations: `n_estimators` × `max_depth` × `learning_rate` × `subsample`
- Best params: `n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8`

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/churn_classifier.ipynb
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
jupyter
```
