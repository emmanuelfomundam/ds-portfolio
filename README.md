# Data Science Portfolio — Emmanuel Fomundam

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-emmanuel--fomundam-blue?logo=linkedin)](https://www.linkedin.com/in/emmanuel-fomundam)
[![GitHub](https://img.shields.io/badge/GitHub-emmanuelfomundam-black?logo=github)](https://github.com/emmanuelfomundam)

End-to-end data science projects spanning data engineering, machine learning, NLP, time series forecasting, and recommendation systems. Every project covers the full lifecycle: raw data ingestion, cleaning, feature engineering, modeling, evaluation, and (where applicable) deployment.

---

## Projects

| # | Project | Techniques | Dataset | Key Result |
|---|---------|-----------|---------|------------|
| 01 | [EDA & Data Cleaning Pipeline](#01--eda--data-cleaning-pipeline) | Pandas, OOP Pipeline, Seaborn | IBM HR Analytics (1,470 records) | Fully automated 7-step cleaning workflow |
| 02 | [Customer Churn Classifier](#02--customer-churn-classifier) | XGBoost, SMOTE, GridSearchCV | Telco Churn (7,043 records) | ROC-AUC **0.9344** via 5-fold CV |
| 03 | [Sales Time Series Forecaster](#03--sales-time-series-forecaster) | ARIMA, XGBoost, Lag Features | Rossmann Store Sales (844K records) | RMSE **985** vs 3,608 ARIMA baseline |
| 04 | [Text Sentiment Analyzer](#04--text-sentiment-analyzer) | TF-IDF, NLP, Streamlit | Amazon Fine Food Reviews (100K) | **82% accuracy**, live Streamlit app |
| 05 | [Recommendation Engine](#05--recommendation-engine) | SVD, Collaborative Filtering, Hybrid | MovieLens 1M (32M ratings) | CV RMSE **0.8783**, MAE **0.6647** |

---

## 01 — EDA & Data Cleaning Pipeline

**[→ Project Folder](01_eda_cleaning_pipeline/)**

A production-style, reusable data cleaning and exploratory analysis pipeline built for business reporting use cases.

**What it does:**
- Ingests the IBM HR Analytics Employee Attrition dataset (1,470 records, 35 features)
- Runs a chained 7-step `DataCleaningPipeline` OOP class: column standardization, constant-column removal, deduplication, median/mode null imputation, IQR outlier capping, label encoding, and feature engineering
- Engineers `tenure_bucket` and `salary_band` features from continuous variables
- Generates full EDA: distribution plots, 34-feature correlation heatmap, and attrition breakdowns via Seaborn/Matplotlib

**Tech:** Python, Pandas, NumPy, Seaborn, Matplotlib, scikit-learn

---

## 02 — Customer Churn Classifier

**[→ Project Folder](02_customer_churn_classifier/)**

A binary classification model to predict telecom customer churn, with a focus on handling class imbalance and maximizing recall on the minority (churn) class.

**What it does:**
- Preprocesses 7,043 Telco records (19 features): coerces mixed-type columns, label encodes categoricals, scales with `StandardScaler`
- Applies SMOTE to balance a 5,174 / 1,869 class split — expanding the resampled training set to 8,278 samples
- Trains and benchmarks Logistic Regression, Random Forest, and XGBoost
- Tunes XGBoost via 5-fold stratified `GridSearchCV` across 16 hyperparameter combinations
- Best CV **ROC-AUC: 0.9344** (vs. 0.8390 Logistic Regression baseline)
- Serializes final model with `joblib`; visualizes confusion matrices, ROC curves, and top-15 feature importances

**Tech:** Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Matplotlib, Seaborn, joblib

---

## 03 — Sales Time Series Forecaster

**[→ Project Folder](03_sales_forecaster/)**

A multi-model sales forecasting system comparing an ARIMA statistical baseline against a feature-rich XGBoost regressor on real-world retail data at scale.

**What it does:**
- Merges and filters sales + store metadata into 844,392 open-day records across 1,115 Rossmann stores
- Engineers a 29-feature dataset: 7/14/28-day lag features, 7/28-day rolling means, and temporal flags (year, month, week, weekend, month-start/end, promo, holiday) via store-grouped transformations
- Runs ARIMA(1,1,1) as a weekly baseline on Store 1 — **RMSE: 3,608**
- Trains log-transformed XGBoost (500 estimators, time-based train/test split at 2015-06-01) on 58,611 held-out records
- Final XGBoost: **RMSE: 985**, **MAPE: 10.44%** — a 73% reduction over baseline
- Plots forecast vs. actuals per store and top-15 feature importances

**Tech:** Python, Pandas, NumPy, XGBoost, Statsmodels, Matplotlib, scikit-learn

---

## 04 — Text Sentiment Analyzer

**[→ Project Folder](04_sentiment_analyzer/)**

A 3-class sentiment classifier (positive / neutral / negative) trained on Amazon product reviews, with a live Streamlit web app for real-time inference.

**What it does:**
- Samples 100,000 reviews from the Amazon Fine Food Reviews dataset; maps 1–2 star → negative, 3 star → neutral, 4–5 star → positive
- Cleans text via regex (HTML stripping, lowercasing, punctuation removal) and NLTK stopword filtering
- Vectorizes with TF-IDF (30,000 features, unigram + bigram range)
- Trains and evaluates Logistic Regression and Naive Bayes baselines
- Best model: **82% overall accuracy**, **0.90 F1** on the dominant positive class
- Serializes model + vectorizer pipeline with `joblib`; deploys as a Streamlit app with live confidence scoring (up to 99.95% on clear-cut reviews)

**Tech:** Python, NLTK, scikit-learn, TF-IDF, Pandas, Streamlit, Matplotlib, joblib, WordCloud

---

## 05 — Recommendation Engine

**[→ Project Folder](05_recommendation_engine/)**

A four-mode hybrid recommendation system built on MovieLens data, combining collaborative filtering, content-based filtering, SVD matrix factorization, and a cold-start popularity fallback.

**What it does:**
- Loads 32M MovieLens ratings across a 2,000-user / 17,338-movie sparse matrix (99.08% sparsity)
- Implements **user-based CF** and **item-based CF** via cosine similarity
- Builds a **content-based filter** using TF-IDF (500 features) on movie genres
- Trains **SVD** (50 latent factors) with the Surprise library: 5-fold CV **RMSE: 0.8783**, **MAE: 0.6647**
- Fuses SVD and content signals into a **weighted hybrid recommender** (70/30 split, min-max normalized) with a popularity fallback for cold-start users (50+ ratings threshold)

**Tech:** Python, Pandas, NumPy, scikit-learn, Surprise (SVD), Matplotlib, Seaborn

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python 3.10+, SQL |
| ML / Modeling | scikit-learn, XGBoost, Statsmodels, Surprise |
| NLP | NLTK, TF-IDF, WordCloud |
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |
| Dev Tools | Git, Jupyter, VS Code, Black |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/emmanuelfomundam/ds-portfolio.git
cd ds-portfolio

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies for a specific project
cd 02_customer_churn_classifier
pip install -r requirements.txt
```

Each project folder contains its own `requirements.txt` and `README.md`.

---

## Author

**Emmanuel Fomundam**
M.S. Data Science — University of Pittsburgh
B.S. Computer Information Systems — Westminster College

📧 emmanuelfomundam12@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/emmanuel-fomundam)
🐙 [GitHub](https://github.com/emmanuelfomundam)
