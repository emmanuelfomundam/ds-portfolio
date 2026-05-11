# 03 — Sales Time Series Forecaster

## Overview

A multi-model sales forecasting system that benchmarks an ARIMA statistical baseline against a feature-engineered XGBoost regressor on large-scale retail data. Achieved a **73% reduction in RMSE** over the ARIMA baseline (3,608 → 985) with a final **MAPE of 10.44%** on 58,611 held-out records.

## Dataset

**Rossmann Store Sales**
- 844,392 open-day records across 1,115 stores
- Features: sales, promotions, state/school holidays, store type, competition distance
- Source: [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data)

> Download `train.csv` and `store.csv` and place them in `data/raw/`.

## Project Structure

```
03_sales_forecaster/
├── data/
│   ├── raw/                        # Raw CSVs (not tracked by git)
│   └── processed/                  # Engineered features (not tracked by git)
├── notebooks/
│   └── sales_forecasting.ipynb     # Full walkthrough notebook
├── src/
│   ├── feature_engineering.py      # load_and_merge(), engineer_features()
│   └── forecast.py                 # encode_cats(), time_split(),
│                                   # train_xgboost(), evaluate(),
│                                   # plot_feature_importance(),
│                                   # run_arima_baseline()
├── outputs/
│   ├── forecast_vs_actuals.png
│   ├── arima_baseline.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Methodology

### Feature Engineering (29 features)
| Category | Features |
|----------|----------|
| Temporal | Year, Month, Week, DayOfWeek, IsWeekend, IsMonthStart, IsMonthEnd |
| Lag | Sales_lag_7, Sales_lag_14, Sales_lag_28 |
| Rolling | Sales_roll_7, Sales_roll_28 |
| Store | StoreType, Assortment, CompetitionDistance |
| Event | Promo, StateHoliday, SchoolHoliday |

### Models
| Model | RMSE | MAPE |
|-------|------|------|
| ARIMA(1,1,1) baseline (Store 1, weekly) | 3,608 | — |
| **XGBoost (log-transformed, 500 estimators)** | **985** | **10.44%** |

### Train/Test Split
- Time-based cutoff at `2015-06-01`
- Train: ~785K records | Test: 58,611 records
- No data leakage — all lag/rolling features use `.shift(1)` before rolling

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sales_forecasting.ipynb
```

## Requirements

```
pandas
numpy
xgboost
statsmodels
scikit-learn
matplotlib
jupyter
```
