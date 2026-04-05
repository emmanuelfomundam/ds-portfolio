import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import statsmodels.api as sm


FEATURE_COLS = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance',
    'Year', 'Month', 'Week', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
    'Sales_lag_7', 'Sales_lag_14', 'Sales_lag_28',
    'Sales_roll_7', 'Sales_roll_28'
]


def encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    for col in ['StateHoliday', 'StoreType', 'Assortment']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def time_split(df: pd.DataFrame, cutoff_date='2015-06-01'):
    train = df[df['Date'] < cutoff_date].copy()
    test  = df[df['Date'] >= cutoff_date].copy()
    print(f'Train: {train.shape} | Test: {test.shape}')
    return train, test


def train_xgboost(train: pd.DataFrame):
    X_tr = train[FEATURE_COLS]
    y_tr = np.log1p(train['Sales'])
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr, verbose=False)
    print('XGBoost model trained.')
    return model


def evaluate(model, test: pd.DataFrame, output_dir='outputs'):
    X_te  = test[FEATURE_COLS]
    y_te  = test['Sales']
    preds = np.expm1(model.predict(X_te))
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mape  = mean_absolute_percentage_error(y_te, preds) * 100
    print(f'Test RMSE : {rmse:,.2f}')
    print(f'Test MAPE : {mape:.2f}%')

    store_id = test['Store'].iloc[0]
    mask = test['Store'] == store_id
    subset = test[mask].copy()
    subset['Predicted'] = preds[mask.values]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(subset['Date'], subset['Sales'],     label='Actual',    linewidth=1.5)
    ax.plot(subset['Date'], subset['Predicted'], label='Predicted',
            linewidth=1.5, linestyle='--', color='orange')
    ax.set_title(f'Forecast vs Actuals - Store {store_id}', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_vs_actuals.png', dpi=150)
    plt.show()
    return rmse, mape


def plot_feature_importance(model, top_n=15, output_dir='outputs'):
    scores = model.feature_importances_
    idx    = np.argsort(scores)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(FEATURE_COLS)[idx], scores[idx], color='#2ecc71')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
    plt.show()


def run_arima_baseline(df: pd.DataFrame, store_id=1, output_dir='outputs'):
    store_df = (
        df[df['Store'] == store_id]
        .set_index('Date')['Sales']
        .resample('W').sum()
    )
    train_s = store_df.iloc[:-12]
    test_s  = store_df.iloc[-12:]
    model   = sm.tsa.ARIMA(train_s, order=(1, 1, 1))
    result  = model.fit()
    forecast = result.forecast(steps=12)
    rmse = np.sqrt(mean_squared_error(test_s, forecast))
    print(f'ARIMA Baseline RMSE (Store {store_id}): {rmse:,.2f}')

    plt.figure(figsize=(12, 4))
    train_s.plot(label='Train')
    test_s.plot(label='Actual')
    forecast.plot(label='ARIMA Forecast', linestyle='--', color='red')
    plt.title(f'ARIMA Baseline - Store {store_id} (Weekly)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/arima_baseline.png', dpi=150)
    plt.show()
