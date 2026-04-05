import pandas as pd
import numpy as np


class DataCleaningPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {}

    def standardize_columns(self):
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w]', '_', regex=True)
        )
        return self

    def drop_constants(self):
        const_cols = [c for c in self.df.columns if self.df[c].nunique() <= 1]
        self.df.drop(columns=const_cols, inplace=True)
        self.report['dropped_constant_cols'] = const_cols
        print(f'Dropped {len(const_cols)} constant columns: {const_cols}')
        return self

    def drop_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = before - len(self.df)
        self.report['duplicates_removed'] = removed
        print(f'Removed {removed} duplicate rows.')
        return self

    def handle_nulls(self, strategy='median'):
        null_summary = self.df.isnull().sum()
        null_cols = null_summary[null_summary > 0]
        self.report['null_cols'] = null_cols.to_dict()
        for col in null_cols.index:
            if self.df[col].dtype in ['float64', 'int64']:
                fill_val = self.df[col].median() if strategy == 'median' else self.df[col].mean()
                self.df[col].fillna(fill_val, inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print(f'Imputed nulls in {len(null_cols)} columns.')
        return self

    def remove_outliers_iqr(self, cols: list):
        for col in cols:
            if col not in self.df.columns:
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            self.df[col] = self.df[col].clip(lower, upper)
            print(f'{col}: capped {outliers} outliers.')
        return self

    def encode_categoricals(self, cols: list):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in cols:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        print(f'Encoded {len(cols)} categorical columns.')
        return self

    def engineer_features(self):
        if 'totalworkingyears' in self.df.columns:
            self.df['tenure_bucket'] = pd.cut(
                self.df['totalworkingyears'],
                bins=[0, 5, 10, 20, 40],
                labels=['0-5', '5-10', '10-20', '20+']
            )
        if 'monthlyincome' in self.df.columns:
            self.df['salary_band'] = pd.cut(
                self.df['monthlyincome'],
                bins=[0, 3000, 6000, 10000, 20000],
                labels=['Low', 'Mid', 'High', 'Executive']
            )
        print('Feature engineering complete.')
        return self

    def get_clean_data(self) -> pd.DataFrame:
        return self.df

    def get_report(self) -> dict:
        return self.report
