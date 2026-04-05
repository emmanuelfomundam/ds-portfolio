import pandas as pd
import numpy as np


def load_and_merge(sales_path: str, store_path: str) -> pd.DataFrame:
    sales = pd.read_csv(sales_path, low_memory=False)
    store = pd.read_csv(store_path)
    df = sales.merge(store, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(['Store', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df['Open'] == 1].copy()
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df.fillna(0, inplace=True)
    store_count = df['Store'].nunique()
    print(f'Loaded {len(df):,} open-day records across {store_count} stores.')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Year']         = df['Date'].dt.year
    df['Month']        = df['Date'].dt.month
    df['Week']         = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek']    = df['Date'].dt.dayofweek
    df['IsWeekend']    = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd']   = df['Date'].dt.is_month_end.astype(int)

    df.sort_values(['Store', 'Date'], inplace=True)
    df['Sales_lag_7']  = df.groupby('Store')['Sales'].shift(7)
    df['Sales_lag_14'] = df.groupby('Store')['Sales'].shift(14)
    df['Sales_lag_28'] = df.groupby('Store')['Sales'].shift(28)
    df['Sales_roll_7']  = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df['Sales_roll_28'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'Feature engineering complete. Shape: {df.shape}')
    return df
