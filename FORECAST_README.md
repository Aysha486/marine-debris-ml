# Future Debris Forecasting System

## Overview
Generate 3-year ahead predictions for marine debris using trained LightGBM time-series models.

## Features
- **Per-country forecasting** with confidence intervals
- **Interactive visualizations** (Plotly charts, Folium maps)
- **Streamlit dashboard** with forecast UI
- **Batch processing** for multiple countries
- **Performance metrics** from cross-validated evaluations

## Quick Start

### 1. Generate Forecasts (CLI)

```bash
# Forecast top 10 countries
python forecast_future_debris.py --top 10

# Forecast specific countries
python forecast_future_debris.py --countries "UNITED STATES" "PHILIPPINES" "CANADA"

# Forecast all countries with ≥5 years data
python forecast_future_debris.py --all
```

### 2. Create Visualizations

```bash
# Generate interactive dashboard
python forecast_visualizations.py --dashboard

# Create forecast map for specific year
python forecast_visualizations.py --map-year 2026

# Plot single country
python forecast_visualizations.py --country "PHILIPPINES"
```

### 3. Run Streamlit Dashboard

```bash
streamlit run app.py
```

Navigate to the **"Future Debris Forecasting"** section for the interactive UI.

## Output Files

### Forecasts
- `future_forecasts_combined.csv` — combined forecast for all countries
- `future_forecast_<Country>.csv` — per-country forecast CSVs

### Visualizations
- `forecast_dashboard.html` — comprehensive multi-panel dashboard
- `forecast_map.html` — interactive map with forecast overlays
- `forecast_comparison.html` — comparison plot for selected countries
- `forecast_plot_<Country>.html` — individual country trend plot

## Data Format

### Input (Historical)
```csv
year,COUNTRY / LOCATION,PEOPLE,Pounds,MILES,Total Items Collected,...
2013,ARGENTINA,302,8018,1.8,6849,...
2014,ARGENTINA,281,4464,33.6,23825,...
```

### Output (Forecast)
```csv
year,country,predicted_debris,lower_ci,upper_ci
2024,ARGENTINA,15234.56,8456.78,22012.34
2025,ARGENTINA,16890.12,9123.45,24656.79
2026,ARGENTINA,18456.89,9987.65,26926.13
```

## Streamlit Dashboard Features

### 📊 Forecast Generation
- **Top N mode:** Select top countries by historical debris volume
- **Custom mode:** Multi-select specific countries
- **All countries mode:** Forecast all available (≥5 years data)

### 📈 Visualizations
1. **Country Trends:** Historical + forecast line plot with CI bands
2. **Forecast Map:** Interactive geospatial view by year
3. **Comparison:** Multi-country forecast trends
4. **Full Data:** Tabular view with download option

### 📋 Performance Metrics
- Cross-validated MAE/RMSE per country
- Confidence interval ranges
- Historical vs forecast comparison

## Model Details

### Architecture
- **Pipeline:** Preprocessing + LightGBM regressor
- **Features:** Lag features (1-3 years), rolling means/std, YoY change
- **Training:** Per-country time-series split with forward validation

### Confidence Intervals
- Based on historical volatility (±1 std dev)
- Adjusted per country based on CV performance

## Dependencies

```
pandas
numpy
joblib
plotly
folium
streamlit
scikit-learn
lightgbm
```

Install:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### "No forecasts generated"
- Ensure `data_cleaned.csv` exists
- Verify `inference_pipeline.joblib` is trained
- Check countries have ≥3 years historical data

### "Prediction failed"
- Model expects specific feature columns
- Run `build_inference_pipeline.py` to retrain if needed

### "No centroids available"
- Run `generate_hotspot_map_capitals.py` to geocode countries
- Map will still work but without geographic markers

## License
Part of Marine Debris Risk Index (MDRI) project.
