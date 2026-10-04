"""
Simple training script for RandomForest and XGBoost models using `data_cleaned.csv`.
Saves models to `models/` directory as `rf_model.joblib` and `xgb_model.joblib` (if XGBoost available).

Usage:
    python train_models.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = r"data_cleaned.csv"
OUT_DIR = r"models"
TARGET = 'Total Items Collected'

os.makedirs(OUT_DIR, exist_ok=True)


def safe_load():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run imputation.ipynb first.")
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_xy(df):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {DATA_PATH}")
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(float)
    X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors='ignore')
    if X.shape[1] == 0:
        raise ValueError('No numeric features available for modeling')
    return X, y


def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def try_train_xgb(X_train, y_train):
    try:
        import xgboost as xgb
        xgbm = xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        xgbm.fit(X_train, y_train)
        return xgbm
    except Exception as e:
        print('XGBoost not available or failed to train:', e)
        return None


if __name__ == '__main__':
    print('Loading data...')
    df = safe_load()
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training RandomForest...')
    rf = train_rf(X_train, y_train)
    preds = rf.predict(X_test)
    print('RF MAE:', mean_absolute_error(y_test, preds))
    joblib.dump(rf, os.path.join(OUT_DIR, 'rf_model.joblib'))
    print('Saved RF model to', os.path.join(OUT_DIR, 'rf_model.joblib'))

    print('Attempting XGBoost training...')
    xgbm = try_train_xgb(X_train, y_train)
    if xgbm is not None:
        preds_x = xgbm.predict(X_test)
        print('XGBoost MAE:', mean_absolute_error(y_test, preds_x))
        joblib.dump(xgbm, os.path.join(OUT_DIR, 'xgb_model.joblib'))
        print('Saved XGBoost model to', os.path.join(OUT_DIR, 'xgb_model.joblib'))
    else:
        print('XGBoost model not available - skipped')
