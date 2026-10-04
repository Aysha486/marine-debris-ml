import pandas as pd
import os

# Files to normalize
files_to_check = [
    'data_cleaned.csv',
    'country_capital_centroids.csv',
    'data_synthetic_all_countries.csv'
]

def normalize_country_names(file_path, country_column):
    """Normalize country names to uppercase"""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print('='*60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} rows")
        
        # Find country column
        country_cols = [col for col in df.columns if 'country' in col.lower() or 'location' in col.lower()]
        
        if not country_cols:
            print(f"⚠ No country column found. Columns: {df.columns.tolist()}")
            return
        
        actual_col = country_cols[0]
        print(f"✓ Found country column: '{actual_col}'")
        
        # Check for mixed case
        original_values = df[actual_col].astype(str)
        mixed_case = df[original_values != original_values.str.upper()]
        
        print(f"\nBefore normalization:")
        print(f"  - Total unique countries: {df[actual_col].nunique()}")
        print(f"  - Mixed case entries: {len(mixed_case)}")
        
        if len(mixed_case) > 0:
            print(f"  - Sample mixed case: {mixed_case[actual_col].head(10).tolist()}")
        
        # Normalize to uppercase and strip whitespace
        df[actual_col] = df[actual_col].astype(str).str.strip().str.upper()
        
        print(f"\nAfter normalization:")
        print(f"  - Total unique countries: {df[actual_col].nunique()}")
        print(f"  - Sample values: {df[actual_col].unique()[:10].tolist()}")
        
        # Save back
        df.to_csv(file_path, index=False)
        print(f"✓ Saved normalized data to {file_path}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Process each file
print("Starting country name normalization...")

normalize_country_names('data_cleaned.csv', 'COUNTRY / LOCATION')
normalize_country_names('country_capital_centroids.csv', 'country')
normalize_country_names('data_synthetic_all_countries.csv', 'COUNTRY / LOCATION')

print("\n" + "="*60)
print("✅ Country name normalization complete!")
print("="*60)
