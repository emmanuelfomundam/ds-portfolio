import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              ConfusionMatrixDisplay, RocCurveDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def split_and_resample(X, y, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    print(f'After SMOTE - Train: {X_res.shape}, Test: {X_te.shape}')
    return X_res, X_te, y_res, y_te


def train_models(X_tr, y_tr):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        trained[name] = model
        print(f'Trained: {name}')
    return trained


def evaluate_models(trained: dict, X_te, y_te, output_dir='outputs'):
    results = {}
    for name, model in trained.items():
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        results[name] = {'auc': auc}
        print(f'\n-- {name} --')
        print(f'ROC-AUC: {auc:.4f}')
        print(classification_report(y_te, y_pred))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, model) in zip(axes, trained.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_te, y_te, ax=ax, colorbar=False, cmap='Blues'
        )
        ax.set_title(name)
    plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in trained.items():
        RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, name=name)
    ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150)
    plt.show()

    return results


def tune_xgboost(X_tr, y_tr):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(xgb, param_grid, cv=cv,
                      scoring='roc_auc', n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    print(f'Best params: {gs.best_params_}')
    print(f'Best CV AUC: {gs.best_score_:.4f}')
    return gs.best_estimator_


def plot_feature_importance(model, feature_names, output_dir='outputs', top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[idx], importances[idx], color='#4C72B0')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances (XGBoost)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
    plt.show()


def save_model(model, path='models/churn_model.pkl'):
    joblib.dump(model, path)
    print(f'Model saved to {path}')
