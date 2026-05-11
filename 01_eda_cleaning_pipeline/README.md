# 01 — EDA & Data Cleaning Pipeline for Business Reporting

## Overview

A production-style, reusable exploratory data analysis and cleaning pipeline built on the IBM HR Analytics Employee Attrition dataset. Designed to automate the full preprocessing lifecycle in a single chained workflow, with output ready for downstream modeling.

## Dataset

**IBM HR Analytics Employee Attrition Dataset**
- 1,470 employee records, 35 features
- Target: `Attrition` (Yes / No)
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

> Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` and place it in `data/raw/`.

## Project Structure

```
01_eda_cleaning_pipeline/
├── data/
│   ├── raw/                    # Raw CSV (not tracked by git)
│   └── processed/              # Cleaned output (not tracked by git)
├── notebooks/
│   └── eda_cleaning.ipynb      # Full walkthrough notebook
├── src/
│   └── pipeline.py             # DataCleaningPipeline class
├── outputs/
│   └── plots/                  # EDA visualizations
├── requirements.txt
└── README.md
```

## Pipeline Steps

The `DataCleaningPipeline` class chains 7 automated steps:

| Step | Method | Description |
|------|--------|-------------|
| 1 | `standardize_columns()` | Lowercase, strip, replace spaces/special chars with underscores |
| 2 | `drop_constants()` | Remove zero-variance columns |
| 3 | `drop_duplicates()` | Remove exact duplicate rows |
| 4 | `handle_nulls()` | Median imputation for numeric; mode for categorical |
| 5 | `remove_outliers_iqr()` | IQR-based outlier capping (1.5× rule) |
| 6 | `encode_categoricals()` | Label encoding for object columns |
| 7 | `engineer_features()` | `tenure_bucket` and `salary_band` from continuous vars |

## EDA Highlights

- Distribution plots across all numeric features
- Full 34-feature correlation heatmap
- Attrition breakdowns by department, job role, tenure, and salary band
- Visualizations generated with Seaborn and Matplotlib

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/eda_cleaning.ipynb
```

## Requirements

```
pandas
numpy
scikit-learn
seaborn
matplotlib
jupyter
```

## Key Findings

- `OverTime`, `JobLevel`, and `MonthlyIncome` are the strongest attrition predictors
- Employees with under 5 years tenure account for the majority of attrition cases
- `StandardHours` and `EmployeeCount` were dropped as zero-variance constants
