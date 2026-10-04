"""
Unified Deep Learning Forecast Engine
Integrates classical ML (LightGBM) with deep learning models (LSTM, GRU, TCN, Transformer)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from dl_forecasters import (
    get_model, train_model, predict_future,
    TimeSeriesDataset, create_sequences
)


class UnifiedForecastEngine:
    """
    Unified forecasting engine supporting both classical ML and deep learning models.
    """
    
    def __init__(self, data_path='data_cleaned.csv'):
        """
        Initialize with cleaned historical data.
        
        Args:
            data_path: Path to cleaned CSV
        """
        self.data = pd.read_csv(data_path)
        self.target_col = 'Total Items Collected'
        self.scalers = {}  # Per-country scalers
        self.models = {}   # Trained models
        
        # Feature columns (exclude identifiers and target)
        exclude = ['year', 'COUNTRY / LOCATION', self.target_col]
        self.feature_cols = [c for c in self.data.columns 
                            if c not in exclude and not c.endswith('_raw')]
        
        print(f"Loaded {len(self.data)} rows, {len(self.data['COUNTRY / LOCATION'].unique())} countries")
        print(f"Feature columns: {len(self.feature_cols)}")
    
    def prepare_country_data(self, country, seq_length=12):
        """
        Prepare time-series data for a single country.
        
        Args:
            country: Country name
            seq_length: Input sequence length
        
        Returns:
            Scaled sequences, scaler object
        """
        # Filter country data
        country_data = self.data[self.data['COUNTRY / LOCATION'] == country].copy()
        country_data = country_data.sort_values('year')
        
        # Adjust sequence length if needed
        min_required = seq_length + 3
        if len(country_data) < min_required:
            # Try shorter sequence
            adjusted_seq = max(3, len(country_data) - 3)
            if adjusted_seq < 3:
                raise ValueError(f"Insufficient data for {country}: only {len(country_data)} years (need ≥6)")
            print(f"  Adjusting sequence length from {seq_length} to {adjusted_seq} for {country}")
            seq_length = adjusted_seq
        
        # Select features
        feature_data = country_data[self.feature_cols + [self.target_col]].values
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Store scaler
        self.scalers[country] = scaler
        
        return scaled_data, scaler, country_data, seq_length
    
    def train_dl_model(self, country, model_type='lstm', seq_length=12, forecast_horizon=3,
                       epochs=100, batch_size=16, val_split=0.2, device='cpu'):
        """
        Train a deep learning model for a single country.
        
        Args:
            country: Country name
            model_type: 'lstm', 'gru', 'tcn', or 'transformer'
            seq_length: Input sequence length
            forecast_horizon: Prediction horizon
            epochs: Training epochs
            batch_size: Batch size
            val_split: Validation split ratio
            device: 'cpu' or 'cuda'
        
        Returns:
            Trained model, training history, evaluation metrics
        """
        print(f"\nTraining {model_type.upper()} for {country}...")
        
        # Prepare data
        scaled_data, scaler, raw_data, actual_seq_length = self.prepare_country_data(country, seq_length)
        seq_length = actual_seq_length  # Use adjusted length
        
        # Create sequences
        X, y = create_sequences(scaled_data, seq_length, forecast_horizon)
        
        if len(X) < 10:
            raise ValueError(f"Not enough sequences for {country}: {len(X)} samples")
        
        # Train/val split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        model = get_model(model_type, input_size=input_size, forecast_horizon=forecast_horizon)
        
        # Train
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=0.001, device=device
        )
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            predictions = model(X_val_tensor).cpu().numpy()
        
        # Compute metrics
        mae = np.mean(np.abs(predictions - y_val))
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
        
        print(f"  ✓ Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Store model
        model_key = f"{country}_{model_type}"
        self.models[model_key] = {
            'model': model,
            'scaler': scaler,
            'seq_length': seq_length,
            'forecast_horizon': forecast_horizon,
            'history': history,
            'metrics': {'mae': mae, 'rmse': rmse}
        }
        
        return model, history, {'mae': mae, 'rmse': rmse}
    
    def forecast_dl(self, country, model_type='lstm', years_ahead=3, device='cpu'):
        """
        Generate forecast using trained deep learning model.
        
        Args:
            country: Country name
            model_type: Model type
            years_ahead: Years to forecast
            device: 'cpu' or 'cuda'
        
        Returns:
            DataFrame with predictions
        """
        model_key = f"{country}_{model_type}"
        
        if model_key not in self.models:
            raise ValueError(f"Model not trained for {country} with {model_type}")
        
        model_info = self.models[model_key]
        model = model_info['model']
        scaler = model_info['scaler']
        seq_length = model_info['seq_length']
        
        # Get last sequence
        country_data = self.data[self.data['COUNTRY / LOCATION'] == country].sort_values('year')
        last_year = country_data['year'].max()
        
        # Prepare last sequence
        feature_data = country_data[self.feature_cols + [self.target_col]].values
        scaled_data = scaler.transform(feature_data)
        last_sequence = scaled_data[-seq_length:]
        
        # Predict
        predictions = predict_future(model, last_sequence, scaler, 
                                     forecast_horizon=years_ahead, device=device)
        
        # Create forecast dataframe
        forecast_years = [last_year + i + 1 for i in range(years_ahead)]
        forecast_df = pd.DataFrame({
            'year': forecast_years,
            'country': country,
            'model': model_type.upper(),
            'predicted_debris': predictions,
            'lower_ci': predictions * 0.8,  # Simple CI (can be improved)
            'upper_ci': predictions * 1.2
        })
        
        return forecast_df
    
    def compare_models(self, country, models=['lstm', 'gru', 'tcn', 'transformer'],
                      seq_length=12, epochs=50, device='cpu'):
        """
        Train and compare multiple models for a country.
        
        Args:
            country: Country name
            models: List of model types to compare
            seq_length: Sequence length
            epochs: Training epochs
            device: 'cpu' or 'cuda'
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_type in models:
            try:
                _, history, metrics = self.train_dl_model(
                    country, model_type=model_type,
                    seq_length=seq_length, epochs=epochs, device=device
                )
                
                results.append({
                    'country': country,
                    'model': model_type.upper(),
                    'val_mae': metrics['mae'],
                    'val_rmse': metrics['rmse'],
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'epochs_trained': len(history['train_loss'])
                })
                
            except Exception as e:
                print(f"  ✗ Failed to train {model_type} for {country}: {e}")
        
        return pd.DataFrame(results)
    
    def save_models(self, save_dir='dl_models'):
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_key, model_info in self.models.items():
            model_path = os.path.join(save_dir, f'{model_key}.pt')
            torch.save({
                'model_state': model_info['model'].state_dict(),
                'scaler': model_info['scaler'],
                'seq_length': model_info['seq_length'],
                'forecast_horizon': model_info['forecast_horizon'],
                'history': model_info['history'],
                'metrics': model_info['metrics']
            }, model_path)
        
        print(f"✓ Saved {len(self.models)} models to {save_dir}/")
    
    def load_models(self, save_dir='dl_models'):
        """Load pre-trained models."""
        if not os.path.exists(save_dir):
            print(f"Model directory {save_dir} not found")
            return
        
        for filename in os.listdir(save_dir):
            if filename.endswith('.pt'):
                model_path = os.path.join(save_dir, filename)
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Extract model key
                model_key = filename.replace('.pt', '')
                
                # Reconstruct model (requires knowing architecture)
                # This is simplified - production would store model config
                print(f"Loaded {model_key}")
                
                self.models[model_key] = checkpoint
        
        print(f"✓ Loaded {len(self.models)} models from {save_dir}/")


def main():
    """CLI for training and comparing deep learning models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train deep learning forecasters')
    parser.add_argument('--country', type=str, required=True, help='Country to forecast')
    parser.add_argument('--data', type=str, default='data_cleaned.csv', help='Data file path')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['lstm', 'gru', 'tcn', 'transformer'],
                       help='Models to train (lstm, gru, tcn, transformer)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--seq-length', type=int, default=12, help='Sequence length')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    
    args = parser.parse_args()
    
    # Initialize engine with specified data file
    engine = UnifiedForecastEngine(data_path=args.data)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.compare:
        # Compare all models
        print(f"\n{'='*60}")
        print(f"COMPARING MODELS FOR: {args.country}")
        print(f"{'='*60}")
        
        comparison = engine.compare_models(
            args.country,
            models=args.models,
            seq_length=args.seq_length,
            epochs=args.epochs,
            device=args.device
        )
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison.to_string(index=False))
        
        # Save comparison
        comparison.to_csv(f'model_comparison_{args.country}.csv', index=False)
        print(f"\n✓ Saved comparison to model_comparison_{args.country}.csv")
        
        # Generate forecasts with best model (if any trained successfully)
        if not comparison.empty:
            best_model = comparison.loc[comparison['val_mae'].idxmin(), 'model'].lower()
            print(f"\n🏆 Best model: {best_model.upper()} (MAE: {comparison['val_mae'].min():.2f})")
            
            forecast = engine.forecast_dl(args.country, model_type=best_model, device=args.device)
            print("\nForecast:")
            print(forecast.to_string(index=False))
            
            forecast.to_csv(f'dl_forecast_{args.country}_{best_model}.csv', index=False)
        else:
            print("\n✗ No models trained successfully. Check data availability for this country.")
        
    else:
        # Train single model
        model_type = args.models[0]
        model, history, metrics = engine.train_dl_model(
            args.country,
            model_type=model_type,
            seq_length=args.seq_length,
            epochs=args.epochs,
            device=args.device
        )
        
        # Generate forecast
        forecast = engine.forecast_dl(args.country, model_type=model_type, device=args.device)
        
        print("\nForecast:")
        print(forecast.to_string(index=False))
        
        forecast.to_csv(f'dl_forecast_{args.country}_{model_type}.csv', index=False)
    
    # Save models
    engine.save_models()


if __name__ == '__main__':
    main()
