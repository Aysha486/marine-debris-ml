"""
Transfer Learning for Marine Debris Forecasting
Train on aggregate global data, fine-tune on per-country data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from dl_forecasters import LSTMForecaster, GRUForecaster, TimeSeriesDataset, create_sequences, train_model
import joblib
import warnings
warnings.filterwarnings('ignore')

class TransferLearningEngine:
    """
    Transfer learning strategy:
    1. Pre-train on global aggregate data (more samples)
    2. Fine-tune on specific country with limited data
    """
    
    def __init__(self, data_path='data_cleaned.csv'):
        self.df = pd.read_csv(data_path)
        # Use lowercase 'year' column
        self.df['YEAR'] = pd.to_datetime(self.df['year'], format='%Y')
        self.scaler = StandardScaler()
        
    def create_global_dataset(self):
        """
        Aggregate all countries to create global time series
        This gives us more training samples
        """
        print("🌍 Creating Global Aggregate Dataset...")
        
        # Group by year, sum debris across all countries
        global_df = self.df.groupby('YEAR').agg({
            'Total Items Collected': 'sum'
        }).reset_index()
        
        global_df = global_df.sort_values('YEAR')
        
        print(f"   ✓ Global dataset: {len(global_df)} years")
        print(f"   ✓ Year range: {global_df['YEAR'].min().year} - {global_df['YEAR'].max().year}")
        
        return global_df
    
    def create_regional_datasets(self):
        """
        Group countries by region for regional transfer learning
        """
        # Regional groupings (simplified)
        regions = {
            'North America': ['UNITED STATES', 'CANADA', 'MEXICO'],
            'Europe': ['UNITED KINGDOM', 'GERMANY', 'FRANCE', 'SPAIN', 'ITALY'],
            'Asia-Pacific': ['JAPAN', 'CHINA', 'AUSTRALIA', 'SOUTH KOREA', 'INDIA'],
            'Southeast Asia': ['PHILIPPINES', 'THAILAND', 'INDONESIA', 'SINGAPORE'],
            'South America': ['BRAZIL', 'ARGENTINA', 'CHILE'],
            'Middle East': ['SAUDI ARABIA', 'UAE', 'QATAR']
        }
        
        regional_data = {}
        
        for region, countries in regions.items():
            # Filter countries in region
            region_df = self.df[self.df['COUNTRY / LOCATION'].isin(countries)]
            
            if len(region_df) > 0:
                # Aggregate by year
                regional_agg = region_df.groupby('YEAR').agg({
                    'Total Items Collected': 'sum'
                }).reset_index()
                
                regional_data[region] = regional_agg
                print(f"   ✓ {region}: {len(regional_agg)} years, {len(countries)} countries")
        
        return regional_data
    
    def pretrain_on_global(self, model_type='lstm', seq_length=10, epochs=100, device='cpu'):
        """
        Pre-train model on global aggregate data
        """
        print(f"\n🔧 Pre-training {model_type.upper()} on Global Data")
        print("=" * 60)
        
        # Get global data
        global_df = self.create_global_dataset()
        
        # Prepare features
        debris_values = global_df['Total Items Collected'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(debris_values)
        
        # Create sequences
        X, y = create_sequences(scaled_values, seq_length, forecast_horizon=3)
        
        if len(X) < 5:
            print(f"   ✗ Not enough sequences: {len(X)} (need ≥5)")
            return None, None
        
        print(f"   ✓ Created {len(X)} training sequences")
        
        # Split train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        # Initialize model
        input_size = X.shape[2]
        
        if model_type == 'lstm':
            model = LSTMForecaster(input_size=input_size, hidden_size=64, num_layers=2, output_size=3)
        elif model_type == 'gru':
            model = GRUForecaster(input_size=input_size, hidden_size=64, num_layers=2, output_size=3)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device)
        
        # Train
        print(f"\n   Training on {len(train_dataset)} samples, validating on {len(val_dataset)}...")
        trained_model, history = train_model(
            model,
            train_dataset,
            val_dataset,
            epochs=epochs,
            batch_size=8,
            lr=0.001,
            device=device
        )
        
        # Save pre-trained model
        pretrained_path = f'pretrained_{model_type}_global.pth'
        torch.save(trained_model.state_dict(), pretrained_path)
        joblib.dump(self.scaler, f'scaler_{model_type}_global.pkl')
        
        print(f"\n   ✓ Pre-trained model saved: {pretrained_path}")
        print(f"   ✓ Final validation loss: {history['val_loss'][-1]:.4f}")
        
        return trained_model, history
    
    def fine_tune_on_country(self, country, model_type='lstm', pretrained_model=None, 
                            seq_length=5, epochs=30, device='cpu'):
        """
        Fine-tune pre-trained model on specific country
        """
        print(f"\n🎯 Fine-tuning on {country}")
        print("=" * 60)
        
        # Get country data
        country_data = self.df[self.df['COUNTRY / LOCATION'] == country].copy()
        country_data = country_data.sort_values('YEAR')
        
        print(f"   Country data: {len(country_data)} years")
        
        if len(country_data) < 4:
            print(f"   ✗ Not enough data for {country}")
            return None
        
        # Prepare features
        debris_values = country_data['Total Items Collected'].values.reshape(-1, 1)
        scaled_values = self.scaler.transform(debris_values)  # Use global scaler
        
        # Create sequences (smaller seq_length for limited data)
        X, y = create_sequences(scaled_values, seq_length, forecast_horizon=3)
        
        if len(X) < 2:
            print(f"   ✗ Not enough sequences: {len(X)}")
            return None
        
        print(f"   ✓ Created {len(X)} sequences")
        
        # Use all data for fine-tuning (small dataset)
        dataset = TimeSeriesDataset(X, y)
        
        # Load pre-trained model if not provided
        if pretrained_model is None:
            input_size = X.shape[2]
            if model_type == 'lstm':
                pretrained_model = LSTMForecaster(input_size=input_size, hidden_size=64, 
                                                 num_layers=2, output_size=3)
            elif model_type == 'gru':
                pretrained_model = GRUForecaster(input_size=input_size, hidden_size=64,
                                                num_layers=2, output_size=3)
            
            # Load weights
            pretrained_path = f'pretrained_{model_type}_global.pth'
            try:
                pretrained_model.load_state_dict(torch.load(pretrained_path, map_location=device))
                print(f"   ✓ Loaded pre-trained weights from {pretrained_path}")
            except:
                print(f"   ⚠ Could not load pre-trained weights, training from scratch")
        
        pretrained_model = pretrained_model.to(device)
        
        # Fine-tune with lower learning rate
        print(f"\n   Fine-tuning on {len(dataset)} samples...")
        finetuned_model, history = train_model(
            pretrained_model,
            dataset,
            None,  # No validation split (small data)
            epochs=epochs,
            batch_size=4,
            lr=0.0001,  # Lower learning rate for fine-tuning
            device=device,
            patience=10
        )
        
        # Save fine-tuned model
        finetuned_path = f'finetuned_{model_type}_{country}.pth'
        torch.save(finetuned_model.state_dict(), finetuned_path)
        
        print(f"\n   ✓ Fine-tuned model saved: {finetuned_path}")
        print(f"   ✓ Final loss: {history['train_loss'][-1]:.4f}")
        
        return finetuned_model, history


def main():
    """Run transfer learning pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transfer learning for marine debris forecasting')
    parser.add_argument('--pretrain', action='store_true', help='Pre-train on global data')
    parser.add_argument('--finetune', type=str, help='Country to fine-tune on')
    parser.add_argument('--model', choices=['lstm', 'gru'], default='lstm', help='Model type')
    parser.add_argument('--epochs-pretrain', type=int, default=100, help='Pre-training epochs')
    parser.add_argument('--epochs-finetune', type=int, default=30, help='Fine-tuning epochs')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    engine = TransferLearningEngine()
    
    if args.pretrain:
        # Pre-train on global data
        print("🌍 PHASE 1: Pre-training on Global Data")
        print("=" * 60)
        model, history = engine.pretrain_on_global(
            model_type=args.model,
            epochs=args.epochs_pretrain,
            device=args.device
        )
        print("\n✅ Pre-training Complete!")
    
    if args.finetune:
        # Fine-tune on specific country
        print(f"\n🎯 PHASE 2: Fine-tuning on {args.finetune}")
        print("=" * 60)
        model, history = engine.fine_tune_on_country(
            country=args.finetune,
            model_type=args.model,
            epochs=args.epochs_finetune,
            device=args.device
        )
        print("\n✅ Fine-tuning Complete!")
    
    print("\n🎯 Usage Examples:")
    print("   # Pre-train + fine-tune:")
    print("   python transfer_learning.py --pretrain --finetune 'UNITED STATES' --model lstm --epochs-pretrain 100 --epochs-finetune 30")


if __name__ == "__main__":
    main()
