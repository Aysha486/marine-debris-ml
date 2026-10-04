import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

proj = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(proj, 'data_cleaned_normalized.csv')
if not os.path.exists(input_path):
    input_path = os.path.join(proj, 'data_cleaned.csv')

out_dir = proj
per_eval_path = os.path.join(proj, 'per_country_evals.csv')
combined_forecasts = os.path.join(proj, 'top_countries_forecasts_combined_normalized.csv')

print('Loading data from', input_path)
df = pd.read_csv(input_path)

# detect country
country_col = None
for c in df.columns:
    if c.lower() == 'country' or 'country' in c.lower() or 'location' in c.lower():
        country_col = c
        break

if country_col is None:
    raise RuntimeError('Country column not found')

# ensure date
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
elif 'year' in df.columns:
    df['date'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-01-01', errors='coerce')
else:
    print('No year/date column, per-country time-series will use index if numeric year present')

if 'count' not in df.columns and 'Total Items Collected' in df.columns:
    df['count'] = df['Total Items Collected']

# sanitize count
if 'count' in df.columns:
    if df['count'].dtype == object or pd.api.types.is_string_dtype(df['count']):
        s = df['count'].astype(str).str.strip()
        s = s.str.replace(',', '')
        s = s.str.replace('%', '')
        s = s.str.replace(r'[^0-9\.-]', '', regex=True)
        df['count'] = pd.to_numeric(s, errors='coerce')

# group yearly
try:
    country_yearly = df.groupby([df[country_col].fillna('Unknown'), pd.Grouper(key='date', freq='YE')])['count'].sum().reset_index()
except Exception:
    country_yearly = df.groupby([df[country_col].fillna('Unknown'), pd.Grouper(key='date', freq='Y')])['count'].sum().reset_index()

counts = country_yearly.groupby(country_col)['date'].nunique().sort_values(ascending=False)
# choose countries with >=3 years for evaluation/modeling
candidate_countries = counts[counts >= 3].index.tolist()
if not candidate_countries:
    candidate_countries = counts[counts >= 2].index.tolist()
print('Countries selected for modeling:', candidate_countries[:30])

country_forecasts = []
evals = []

for ctry in candidate_countries:
    ser = country_yearly[country_yearly[country_col]==ctry].set_index('date')['count'].dropna().astype(float)
    n_years = ser.shape[0]
    safe_name = re.sub(r'[\\/\s]+', '_', str(ctry))
    if n_years < 2:
        print(f'{ctry}: too few years, skipping')
        continue

    # ensure yearly
    try:
        ser = ser.asfreq('YE').fillna(0)
    except Exception:
        ser = ser.asfreq('Y').fillna(0)

    # create lag features up to 2 lags
    df_l = pd.DataFrame({'y': ser})
    df_l['lag_1'] = df_l['y'].shift(1)
    df_l['lag_2'] = df_l['y'].shift(2)
    df_l.dropna(inplace=True)
    if df_l.shape[0] < 2:
        # carry forward
        last_val = float(ser.dropna().iloc[-1])
        preds = [last_val]*3
        future_idx = pd.date_range(start=ser.index[-1] + pd.offsets.YearEnd(1), periods=3, freq='YE')
        tmp = pd.DataFrame({'date': future_idx, 'forecast': preds, 'country': ctry})
        country_forecasts.append(tmp)
        evals.append({'country': ctry, 'mode': 'carry_forward', 'mae': None, 'rows_train': 0})
        continue

    # train on all but last row, test last
    train = df_l.iloc[:-1]
    test = df_l.iloc[-1:]
    X_train = train.drop(columns=['y'])
    y_train = train['y']
    X_test = test.drop(columns=['y'])
    y_test = test['y']

    # choose model: LGBM if available else HGB
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=42)
        model_name = 'LGBM'
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(random_state=42)
        model_name = 'HGB'

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    evals.append({'country': ctry, 'mode': 'modeled', 'model': model_name, 'mae': float(mae), 'rows_train': X_train.shape[0]})

    # recursive forecast 3 years
    last_vals = ser.dropna().values.tolist()
    preds = []
    for i in range(3):
        # build row: lag_1 = last_vals[-1], lag_2 = last_vals[-2] if exist
        lag_1 = last_vals[-1]
        lag_2 = last_vals[-2] if len(last_vals) >= 2 else last_vals[-1]
        x = pd.DataFrame([[lag_1, lag_2]], columns=['lag_1','lag_2'])
        p = float(model.predict(x)[0])
        preds.append(p)
        last_vals.append(p)

    future_idx = pd.date_range(start=ser.index[-1] + pd.offsets.YearEnd(1), periods=3, freq='YE')
    tmp = pd.DataFrame({'date': future_idx, 'forecast': preds, 'country': ctry})
    country_forecasts.append(tmp)

    # save per-country forecast
    tmp.to_csv(os.path.join(out_dir, f'forecast_{safe_name}.csv'), index=False)

# combine and save
if country_forecasts:
    all_cf = pd.concat(country_forecasts, ignore_index=True)
    all_cf.to_csv(combined_forecasts, index=False)
    print('Saved combined forecasts to', combined_forecasts)

pd.DataFrame(evals).to_csv(per_eval_path, index=False)
print('Saved per-country evaluation summary to', per_eval_path)
print('Done')
