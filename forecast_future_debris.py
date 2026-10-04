"""
Future Debris Forecasting System
Generates 3-year ahead predictions for marine debris per country using existing LightGBM model.
"""
import os
import pandas as pd
import numpy as np
import joblib
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DebrisForecastEngine:
    """Generates future debris forecasts using trained time-series model."""
    
    def __init__(self, data_path: str, model_path: str, synthetic_path: str = 'data_synthetic_all_countries.csv'):
        """
        Initialize forecaster with cleaned data and trained model.
        
        Args:
            data_path: Path to cleaned CSV with historical data
            model_path: Path to trained inference pipeline (.joblib)
            synthetic_path: Optional path to synthetic data for sparse countries
        """
        self.data = pd.read_csv(data_path)
        self.synthetic_data = None
        if os.path.exists(synthetic_path):
            try:
                self.synthetic_data = pd.read_csv(synthetic_path)
            except Exception:
                self.synthetic_data = None
        
        self.model = joblib.load(model_path)
        self.forecast_horizon = 3  # default years
        
        # Get expected feature columns from the pipeline
        try:
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_cols = list(self.model.feature_names_in_)
            else:
                target_col = 'Total Items Collected'
                # Keep 'year' so the model can learn temporal trend
                exclude = ['COUNTRY / LOCATION', target_col]
                self.feature_cols = [c for c in self.data.columns if c not in exclude and not c.endswith('_raw')]
        except Exception:
            target_col = 'Total Items Collected'
            exclude = ['COUNTRY / LOCATION', target_col]
            self.feature_cols = [c for c in self.data.columns if c not in exclude and not c.endswith('_raw')]
        
        print(f"Loaded data: {len(self.data)} rows, {len(self.data['COUNTRY / LOCATION'].unique())} countries")
        if self.synthetic_data is not None:
            print(f"Synthetic fallback available: {len(self.synthetic_data)} rows")
        print(f"Feature columns: {len(self.feature_cols)}")
        print(f"Expected features: {self.feature_cols[:5]}..." if len(self.feature_cols) > 5 else f"Features: {self.feature_cols}")

    def _get_country_data(self, country: str, min_years: int = 3) -> Tuple[pd.DataFrame, str]:
        """Return country data; fall back to synthetic if real data is insufficient."""
        real = self.data[self.data['COUNTRY / LOCATION'] == country].copy()
        source = 'real'
        if len(real) >= min_years:
            return real.sort_values('year'), source
        # try synthetic
        if self.synthetic_data is not None:
            synth = self.synthetic_data[self.synthetic_data['COUNTRY / LOCATION'] == country].copy()
            if not synth.empty:
                # prefer synthetic if it extends history
                combined = pd.concat([real, synth], ignore_index=True)
                combined = combined.drop_duplicates(subset=['year'], keep='last').sort_values('year')
                source = 'synthetic' if len(real) == 0 else 'mixed'
                if len(combined) >= min_years:
                    return combined, source
                real = combined  # fall through for auto-fill below
        # still short: fabricate minimal synthetic points using last known value
        if real.empty:
            baseline = self.data['Total Items Collected'].median()
            last_year = int(self.data['year'].max())
            years = [last_year - 2, last_year - 1, last_year]
            vals = [baseline * 0.9, baseline, baseline * 1.05]
            real = pd.DataFrame({
                'COUNTRY / LOCATION': country,
                'year': years,
                'Total Items Collected': vals
            })
            source = 'synthetic'
        elif len(real) == 1:
            last_year = int(real['year'].iloc[-1])
            last_val = float(real['Total Items Collected'].iloc[-1])
            real = pd.concat([
                real,
                pd.DataFrame({
                    'COUNTRY / LOCATION': [country, country],
                    'year': [last_year - 1, last_year - 2],
                    'Total Items Collected': [last_val * 0.9, last_val * 0.8]
                })
            ], ignore_index=True)
            source = 'mixed'
        return real.sort_values('year'), source
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Total Items Collected') -> pd.DataFrame:
        """Create lag features for time series forecasting."""
        df = df.copy().sort_values('year').reset_index(drop=True)
        
        # Lag features (1, 2, 3 years)
        for lag in [1, 2, 3]:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics (2-year and 3-year windows)
        df[f'{target_col}_rolling_mean_2'] = df[target_col].shift(1).rolling(window=2, min_periods=1).mean()
        df[f'{target_col}_rolling_mean_3'] = df[target_col].shift(1).rolling(window=3, min_periods=1).mean()
        df[f'{target_col}_rolling_std_2'] = df[target_col].shift(1).rolling(window=2, min_periods=1).std()
        
        # Year-over-year change
        df[f'{target_col}_yoy_change'] = df[target_col].diff()
        
        return df
    
    def get_available_countries(self, min_years: int = 1) -> List[str]:
        """Get list of countries, optionally filtered by minimum history."""
        country_counts = self.data.groupby('COUNTRY / LOCATION').size()
        if min_years <= 1:
            return sorted(country_counts.index.tolist())
        valid = country_counts[country_counts >= min_years].index.tolist()
        return sorted(valid)
    
    def forecast_country(self, country: str, include_ci: bool = True, years_ahead: int = None,
                         target_year: int = None, min_years: int = 3) -> pd.DataFrame:
        """
        Generate forecast for a single country. Uses synthetic data if real history is short.
        """
        # Get historical data (with synthetic fallback)
        country_data, source = self._get_country_data(country, min_years=min_years)
        country_data = country_data.sort_values('year')
        
        if len(country_data) < 2:
            raise ValueError(f"Insufficient data for {country} even after synthetic fill")
        
        # Create lag features
        country_data = self.create_lag_features(country_data)
        
        # Determine forecast horizon
        last_year = int(country_data['year'].max())
        if target_year is not None and target_year > last_year:
            horizon = max(1, target_year - last_year)
        else:
            horizon = years_ahead if years_ahead is not None else self.forecast_horizon
        horizon = max(1, int(horizon))
        
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))
        forecasts = []
        current_data = country_data.copy()
        target_col = 'Total Items Collected'
        
        for forecast_year in forecast_years:
            latest = current_data.iloc[-1].copy()
            new_row = latest.copy()
            new_row['year'] = forecast_year
            
            feature_dict = {}
            # Inject future year so the model can create non-flat trajectories
            if 'year' in self.feature_cols:
                feature_dict['year'] = forecast_year
            for col in self.feature_cols:
                if col in latest.index:
                    feature_dict[col] = latest[col]
                else:
                    feature_dict[col] = 0
            
            if f'{target_col}_lag1' in self.feature_cols:
                feature_dict[f'{target_col}_lag1'] = latest[target_col]
            if f'{target_col}_lag2' in self.feature_cols:
                feature_dict[f'{target_col}_lag2'] = latest.get(f'{target_col}_lag1', latest[target_col])
            if f'{target_col}_lag3' in self.feature_cols:
                feature_dict[f'{target_col}_lag3'] = latest.get(f'{target_col}_lag2', latest[target_col])
            
            if f'{target_col}_rolling_mean_2' in self.feature_cols:
                recent_values = current_data[target_col].tail(2).values
                feature_dict[f'{target_col}_rolling_mean_2'] = recent_values.mean()
            if f'{target_col}_rolling_mean_3' in self.feature_cols:
                recent_values = current_data[target_col].tail(3).values
                feature_dict[f'{target_col}_rolling_mean_3'] = recent_values.mean()
            if f'{target_col}_rolling_std_2' in self.feature_cols:
                recent_values = current_data[target_col].tail(2).values
                feature_dict[f'{target_col}_rolling_std_2'] = recent_values.std() if len(recent_values) > 1 else 0
            
            if f'{target_col}_yoy_change' in self.feature_cols and len(current_data) > 1:
                feature_dict[f'{target_col}_yoy_change'] = current_data[target_col].diff().iloc[-1]
            
            X_pred = pd.DataFrame([feature_dict])[self.feature_cols]
            
            try:
                pred = self.model.predict(X_pred)[0]
            except Exception as e:
                print(f"Warning: Prediction failed for {country} year {forecast_year}: {e}")
                pred = latest[target_col]
            
            # Apply floor: prevent unrealistic collapse; use 5% of mean historical value
            hist_mean = country_data[target_col].mean()
            floor = max(1, hist_mean * 0.05)
            pred = max(floor, pred)
            
            if include_ci:
                hist_std = country_data[target_col].std()
                lower = max(0, pred - hist_std)
                upper = pred + hist_std
            else:
                lower = upper = pred
            
            forecasts.append({
                'year': forecast_year,
                'country': country,
                'predicted_debris': round(pred, 2),
                'lower_ci': round(lower, 2),
                'upper_ci': round(upper, 2),
                'data_source': source
            })
            
            new_row[target_col] = pred
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            current_data = self.create_lag_features(current_data)
        
        return pd.DataFrame(forecasts)
    
    def forecast_multiple_countries(self, countries: List[str], save_individual: bool = True,
                                     years_ahead: int = None, target_year: int = None,
                                     min_years: int = 3) -> pd.DataFrame:
        """Generate forecasts for multiple countries."""
        all_forecasts = []
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
        
        for i, country in enumerate(countries, 1):
            try:
                print(f"[{i}/{len(countries)}] Forecasting {country}...")
                forecast = self.forecast_country(
                    country,
                    years_ahead=years_ahead,
                    target_year=target_year,
                    min_years=min_years
                )
                all_forecasts.append(forecast)
                
                if save_individual:
                    safe_name = country.replace(' ', '_').replace('/', '_')
                    out_path = os.path.join(output_dir, f'future_forecast_{safe_name}.csv')
                    forecast.to_csv(out_path, index=False)
                    print(f"  → Saved to {out_path}")
                    
            except Exception as e:
                print(f"  ✗ Failed for {country}: {e}")
        
        if all_forecasts:
            combined = pd.concat(all_forecasts, ignore_index=True)
            combined_path = os.path.join(output_dir, 'future_forecasts_combined.csv')
            combined.to_csv(combined_path, index=False)
            print(f"\n✓ Combined forecast saved: {combined_path} ({len(combined)} rows)")
            return combined
        print("\n✗ No forecasts generated")
        return pd.DataFrame()
    
    def get_historical_performance(self, country: str) -> Dict:
        """Get historical CV performance metrics for a country."""
        cv_path = 'per_country_cv_evals.csv'
        if not os.path.exists(cv_path):
            return {}
        
        cv_df = pd.read_csv(cv_path)
        country_cv = cv_df[cv_df['country'] == country]
        
        if len(country_cv) == 0:
            return {}
        
        row = country_cv.iloc[0]
        return {
            'cv_mae_mean': row.get('cv_mae_mean', np.nan),
            'cv_mae_std': row.get('cv_mae_std', np.nan),
            'cv_rmse_mean': row.get('cv_rmse_mean', np.nan),
            'n_folds': row.get('n_folds', 0)
        }


def main():
    """CLI interface for generating forecasts."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate future debris forecasts')
    parser.add_argument('--countries', type=str, nargs='+', help='List of countries to forecast')
    parser.add_argument('--all', action='store_true', help='Forecast all countries with >=5 years data')
    parser.add_argument('--top', type=int, help='Forecast top N countries by total debris')
    parser.add_argument('--data', type=str, default='data_cleaned.csv', help='Path to cleaned data CSV')
    parser.add_argument('--model', type=str, default='inference_pipeline.joblib', help='Path to trained model')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = DebrisForecastEngine(args.data, args.model)
    
    # Determine which countries to forecast
    if args.all:
        countries = engine.get_available_countries(min_years=5)
        print(f"Forecasting all {len(countries)} countries with >=5 years data")
    elif args.top:
        # Get top N by total debris
        country_totals = engine.data.groupby('COUNTRY / LOCATION')['Total Items Collected'].sum()
        top_countries = country_totals.nlargest(args.top).index.tolist()
        countries = [c for c in top_countries if c in engine.get_available_countries()]
        print(f"Forecasting top {len(countries)} countries by debris volume")
    elif args.countries:
        countries = args.countries
        print(f"Forecasting {len(countries)} specified countries")
    else:
        # Default: top 20
        country_totals = engine.data.groupby('COUNTRY / LOCATION')['Total Items Collected'].sum()
        top_countries = country_totals.nlargest(20).index.tolist()
        countries = [c for c in top_countries if c in engine.get_available_countries()]
        print(f"Forecasting top {len(countries)} countries (default)")
    
    # Generate forecasts
    combined = engine.forecast_multiple_countries(countries, save_individual=True)
    
    if not combined.empty:
        print("\n" + "="*60)
        print("FORECAST SUMMARY")
        print("="*60)
        print(f"Countries forecasted: {combined['country'].nunique()}")
        print(f"Years forecasted: {sorted(combined['year'].unique())}")
        
        # Use args.top if provided, otherwise default to 10
        top_n_display = args.top if args.top else 10
        print(f"\nTop {top_n_display} predicted debris volumes (final year):")
        final_year = combined['year'].max()
        top10 = combined[combined['year'] == final_year].nlargest(top_n_display, 'predicted_debris')
        print(top10[['country', 'predicted_debris', 'lower_ci', 'upper_ci']].to_string(index=False))


if __name__ == '__main__':
    main()
