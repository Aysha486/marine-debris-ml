import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
import joblib

proj = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(proj, 'data_cleaned_normalized.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(proj, 'data_cleaned.csv')

print('Loading data from', data_path)
df = pd.read_csv(data_path)
TARGET = 'Total Items Collected'
if TARGET not in df.columns:
    raise RuntimeError(f"Target {TARGET} not found in {data_path}")

# basic preprocessing columns
raw_cols = [c for c in df.columns if c.endswith('_raw')]
X = df.drop(columns=raw_cols + [TARGET], errors='ignore').copy()
y = pd.to_numeric(df[TARGET], errors='coerce')
mask = ~y.isna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print('Numeric cols:', numeric_cols[:10])
print('Categorical cols:', categorical_cols[:10])

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols), ('cat', cat_transformer, categorical_cols)])

# choose estimator: try XGBoost -> LightGBM -> RandomForest
try:
    from xgboost import XGBRegressor
    estimator = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    est_name = 'XGB'
except Exception:
    try:
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        est_name = 'LGBM'
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        est_name = 'RF'

pipe = Pipeline(steps=[('pre', preprocessor), ('model', estimator)])

# quick cross-validated evaluation (5-fold) on entire data
print('Running 5-fold CV (neg MAE) with estimator:', est_name)
scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
mae_cv = -scores.mean()
print(f'CV MAE (5-fold): {mae_cv:.4f}')

# fit final pipeline on full data and save
pipe.fit(X, y)
out_path = os.path.join(proj, 'inference_pipeline.joblib')
joblib.dump(pipe, out_path)
print('Saved inference pipeline to', out_path)

# save CV result to CSV
import csv
with open(os.path.join(proj, 'inference_pipeline_eval.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['estimator', 'cv_mae'])
    w.writerow([est_name, float(mae_cv)])

print('Done')
