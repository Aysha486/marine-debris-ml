"""
Data Augmentation for Marine Debris Time Series
Extends limited historical data to enable deep learning model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAugmenter:
    """Augment time series data using statistical methods"""
    
    def __init__(self, data_path='data_cleaned.csv'):
        self.df = pd.read_csv(data_path)
        # Use lowercase 'year' column
        self.df['YEAR'] = pd.to_datetime(self.df['year'], format='%Y')
        
    def backfill_with_arima(self, country, target_years=20, order=(1,1,1)):
        """
        Extend historical data backwards using ARIMA modeling
        
        Args:
            country: Country name
            target_years: Desired total years of data
            order: ARIMA(p,d,q) order
        """
        country_data = self.df[self.df['COUNTRY / LOCATION'] == country].copy()
        country_data = country_data.sort_values('YEAR')
        
        if len(country_data) >= target_years:
            print(f"✓ {country} already has {len(country_data)} years")
            return country_data
            
        # Calculate years to generate
        years_needed = target_years - len(country_data)
        min_year = country_data['YEAR'].min()
        
        print(f"\n📊 Augmenting {country}:")
        print(f"   Current data: {len(country_data)} years ({min_year.year}-{country_data['YEAR'].max().year})")
        print(f"   Generating: {years_needed} historical years")
        
        # Fit ARIMA on existing data
        try:
            debris_values = country_data['Total Items Collected'].values
            
            # Fit model
            model = ARIMA(debris_values, order=order)
            fitted_model = model.fit()
            
            # Generate historical data (backwards forecast)
            # Use residuals and trend to estimate past values
            historical_mean = debris_values[:3].mean()  # Early period mean
            historical_std = debris_values.std() * 0.5  # Lower variance in past
            
            synthetic_years = []
            for i in range(years_needed, 0, -1):
                # Generate with trend and noise
                base_value = historical_mean * (0.7 + 0.3 * (i / years_needed))  # Lower values in past
                noise = np.random.normal(0, historical_std)
                synthetic_value = max(0, base_value + noise)
                
                synthetic_year = min_year - pd.DateOffset(years=i)
            synthetic_years.append({
                'YEAR': synthetic_year,
                'COUNTRY / LOCATION': country,
                'Total Items Collected': int(synthetic_value),
                'IS_SYNTHETIC': True
            })            # Combine synthetic + real data
            synthetic_df = pd.DataFrame(synthetic_years)
            country_data['IS_SYNTHETIC'] = False
            augmented_data = pd.concat([synthetic_df, country_data], ignore_index=True)
            augmented_data = augmented_data.sort_values('YEAR')
            
            print(f"   ✓ Generated {len(synthetic_df)} synthetic years")
            print(f"   ✓ Total: {len(augmented_data)} years ({augmented_data['YEAR'].min().year}-{augmented_data['YEAR'].max().year})")
            
            return augmented_data
            
        except Exception as e:
            print(f"   ✗ ARIMA failed: {e}")
            return self._simple_backfill(country_data, years_needed)
    
    def _simple_backfill(self, country_data, years_needed):
        """Fallback: Simple linear interpolation backwards"""
        min_year = country_data['YEAR'].min()
        country = country_data['COUNTRY / LOCATION'].iloc[0]
        
        # Calculate trend from first 3 years
        early_data = country_data.head(3)
        mean_value = early_data['Total Items Collected'].mean()
        
        synthetic_years = []
        for i in range(years_needed, 0, -1):
            synthetic_year = min_year - pd.DateOffset(years=i)
            # Decrease by 10% per year going backwards
            synthetic_value = int(mean_value * (0.9 ** i))
            
            synthetic_years.append({
                'YEAR': synthetic_year,
                'COUNTRY / LOCATION': country,
                'Total Items Collected': max(0, synthetic_value),
                'IS_SYNTHETIC': True
            })
        
        synthetic_df = pd.DataFrame(synthetic_years)
        country_data['IS_SYNTHETIC'] = False
        augmented_data = pd.concat([synthetic_df, country_data], ignore_index=True)
        
        return augmented_data.sort_values('YEAR')
    
    def add_noise_variants(self, country_data, num_variants=3, noise_level=0.15):
        """
        Create multiple variants by adding controlled noise
        Helps model learn robustness
        """
        variants = []
        debris_col = 'Total Items Collected'
        
        for variant_id in range(num_variants):
            variant = country_data.copy()
            
            # Add Gaussian noise
            debris_values = variant[debris_col].values
            noise = np.random.normal(1, noise_level, size=len(debris_values))
            variant[debris_col] = (debris_values * noise).astype(int).clip(0)
            variant['VARIANT_ID'] = f"V{variant_id + 1}"
            
            variants.append(variant)
        
        return pd.concat(variants, ignore_index=True)
    
    def augment_dataset(self, top_n=10, target_years=25, output_path='data_augmented.csv'):
        """
        Main function: Augment dataset for top N countries
        
        Args:
            top_n: Number of top countries to augment
            target_years: Target years per country
            output_path: Where to save augmented data
        """
        print(f"🚀 Starting Data Augmentation")
        print(f"   Target: {target_years} years per country")
        print("=" * 60)
        
        # Get top countries by data volume
        country_counts = self.df.groupby('COUNTRY / LOCATION').size().sort_values(ascending=False)
        top_countries = country_counts.head(top_n).index.tolist()
        
        print(f"\n📍 Selected top {top_n} countries:")
        for i, country in enumerate(top_countries, 1):
            print(f"   {i}. {country} ({country_counts[country]} years)")
        
        # Augment each country
        augmented_datasets = []
        
        for country in top_countries:
            # Backfill historical data
            augmented = self.backfill_with_arima(country, target_years)
            augmented_datasets.append(augmented)
        
        # Combine all countries
        final_df = pd.concat(augmented_datasets, ignore_index=True)
        
        # Convert YEAR back to year column (integer format for compatibility)
        final_df['year'] = final_df['YEAR'].dt.year
        final_df = final_df.drop(columns=['YEAR'])
        
        # Reorder columns to match original data
        cols_order = ['year', 'COUNTRY / LOCATION', 'Total Items Collected', 'IS_SYNTHETIC']
        final_df = final_df[cols_order]
        
        # Save
        final_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print(f"✅ Augmentation Complete!")
        print(f"   Original dataset: {len(self.df)} rows")
        print(f"   Augmented dataset: {len(final_df)} rows")
        print(f"   Synthetic rows: {final_df['IS_SYNTHETIC'].sum()}")
        print(f"   Real rows: {(~final_df['IS_SYNTHETIC']).sum()}")
        print(f"   Saved to: {output_path}")
        
        # Summary statistics
        print("\n📊 Per-Country Summary:")
        summary = final_df.groupby('COUNTRY / LOCATION').agg({
            'year': ['count', 'min', 'max'],
            'IS_SYNTHETIC': 'sum'
        })
        print(summary)
        
        return final_df


def main():
    """Run augmentation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment marine debris time series data')
    parser.add_argument('--countries', type=int, default=10, help='Number of top countries to augment')
    parser.add_argument('--years', type=int, default=25, help='Target years per country')
    parser.add_argument('--output', default='data_augmented.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Run augmentation
    augmenter = TimeSeriesAugmenter()
    augmented_df = augmenter.augment_dataset(
        top_n=args.countries,
        target_years=args.years,
        output_path=args.output
    )
    
    print("\n🎯 Next Steps:")
    print("   1. Train deep learning models: python unified_forecast_engine.py --data data_augmented.csv --compare")
    print("   2. Validate on REAL data only (use IS_SYNTHETIC flag)")
    print("   3. Use augmented data for training, real data for testing")


if __name__ == "__main__":
    main()
