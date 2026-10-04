"""
Train Deep Learning Models on Synthetic Data
Smart country selection based on data availability
"""
import pandas as pd
import subprocess
import os

print('=' * 80)
print('🚀 TRAINING DEEP LEARNING MODELS ON SYNTHETIC DATA')
print('=' * 80)

# Load and analyze data
df = pd.read_csv('data_synthetic_all_countries.csv')

# Group by country and count real data points
country_real_counts = {}
for country in df['COUNTRY / LOCATION'].unique():
    country_df = df[df['COUNTRY / LOCATION'] == country]
    real_count = (~country_df['IS_SYNTHETIC']).sum()
    total_count = len(country_df)
    if real_count >= 4:  # Need at least 4 real points for meaningful training
        country_real_counts[country] = {
            'real': real_count,
            'total': total_count,
            'synthetic': total_count - real_count
        }

# Sort by total data points
sorted_countries = sorted(country_real_counts.items(), 
                         key=lambda x: x[1]['total'], reverse=True)

print(f'\n📊 Countries with ≥4 Real Data Points:')
print(f'   Total: {len(sorted_countries)} countries')
print()

# Select top 15 for training
top_countries = sorted_countries[:15]

print('🎯 Top 15 Countries for Training:')
print('-' * 80)
for i, (country, counts) in enumerate(top_countries, 1):
    print(f'{i:2d}. {country:30s} - {counts["total"]:3d} total ({counts["real"]} real + {counts["synthetic"]} synthetic)')

# Train models
print('\n' + '=' * 80)
print('⏳ Starting Training...')
print('=' * 80)

results = []

for idx, (country, counts) in enumerate(top_countries, 1):
    print(f'\n[{idx}/15] {country}... ', end='', flush=True)
    
    cmd = [
        'python', 'unified_forecast_engine.py',
        '--data', 'data_synthetic_all_countries.csv',
        '--country', country,
        '--models', 'lstm',
        '--epochs', '30'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print('✅ TRAINED')
            results.append({
                'Country': country,
                'Real Data': counts['real'],
                'Total Data': counts['total'],
                'Status': 'Success'
            })
        else:
            print('❌ FAILED')
            results.append({
                'Country': country,
                'Real Data': counts['real'],
                'Total Data': counts['total'],
                'Status': 'Failed'
            })
    
    except subprocess.TimeoutExpired:
        print('⏱️  TIMEOUT')
        results.append({
            'Country': country,
            'Real Data': counts['real'],
            'Total Data': counts['total'],
            'Status': 'Timeout'
        })
    except Exception as e:
        print(f'❌ ERROR')
        results.append({
            'Country': country,
            'Real Data': counts['real'],
            'Total Data': counts['total'],
            'Status': f'Error: {str(e)[:20]}'
        })

# Summary
print('\n' + '=' * 80)
print('📊 TRAINING SUMMARY')
print('=' * 80)

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# Save results
summary_df.to_csv('training_results.csv', index=False)
print(f'\n✅ Results saved to: training_results.csv')

# Count successful trainings
successful = (summary_df['Status'] == 'Success').sum()
print(f'\n🎉 Successfully trained: {successful}/{len(results)} countries')

# List generated forecast files
print(f'\n📁 Generated Files:')
forecast_files = [f for f in os.listdir('.') if f.startswith('dl_forecast_') and f.endswith('.csv')]
comparison_files = [f for f in os.listdir('.') if f.startswith('model_comparison_') and f.endswith('.csv')]

print(f'   - Forecasts: {len(forecast_files)} files')
print(f'   - Comparisons: {len(comparison_files)} files')

if forecast_files:
    print(f'\n   Sample forecast files:')
    for f in forecast_files[:5]:
        print(f'     {f}')

print('\n' + '=' * 80)
print('🎉 TRAINING COMPLETE!')
print('=' * 80)
