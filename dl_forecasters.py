"""
Deep Learning Time-Series Forecasters
Implements LSTM, GRU, TCN, and Transformer models for marine debris forecasting.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA PREPARATION ====================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time-series data."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(data, seq_length=12, forecast_horizon=3):
    """
    Create sequences for time-series forecasting.
    
    Args:
        data: 2D array (time_steps, features)
        seq_length: Input sequence length
        forecast_horizon: Number of steps to predict ahead
    
    Returns:
        X (sequences), y (targets)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon, 0])  # Target is first column
    return np.array(X), np.array(y)


# ==================== LSTM/GRU MODELS ====================

class LSTMForecaster(nn.Module):
    """LSTM-based time-series forecaster."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_horizon=3):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_out)
        return output


class GRUForecaster(nn.Module):
    """GRU-based time-series forecaster."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_horizon=3):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output


# ==================== TEMPORAL CONVOLUTIONAL NETWORK ====================

class Chomp1d(nn.Module):
    """Remove padding from temporal convolution."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single temporal block for TCN."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network for time-series forecasting."""
    
    def __init__(self, input_size, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2, forecast_horizon=3):
        super(TCNForecaster, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], forecast_horizon)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # TCN forward
        out = self.network(x)
        
        # Global average pooling
        out = out.mean(dim=2)
        
        # Forecast
        output = self.fc(out)
        return output


# ==================== TRANSFORMER MODEL ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerForecaster(nn.Module):
    """Transformer-based time-series forecaster."""
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2, forecast_horizon=3):
        super(TransformerForecaster, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )
    
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        out = self.transformer_encoder(x)
        
        # Use last time step
        last_out = out[:, -1, :]
        
        # Forecast
        output = self.fc(last_out)
        return output


# ==================== TRAINING UTILITIES ====================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """
    Train a deep learning forecaster.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained model, training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=15)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return model, history


def predict_future(model, last_sequence, scaler, forecast_horizon=3, device='cpu'):
    """
    Generate future predictions using trained model.
    
    Args:
        model: Trained PyTorch model
        last_sequence: Last known sequence (seq_len, features)
        scaler: StandardScaler for inverse transform
        forecast_horizon: Steps to predict
        device: 'cpu' or 'cuda'
    
    Returns:
        Array of predictions
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        X = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        # Predict
        output = model(X)
        
        # Convert to numpy
        predictions = output.cpu().numpy().flatten()
        
        # Inverse transform (assuming target is first column)
        # Create dummy array with same shape as original features
        dummy = np.zeros((forecast_horizon, last_sequence.shape[1]))
        dummy[:, 0] = predictions
        
        # Inverse transform
        predictions_original = scaler.inverse_transform(dummy)[:, 0]
        
        return predictions_original


# ==================== MODEL FACTORY ====================

def get_model(model_type, input_size, forecast_horizon=3, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: 'lstm', 'gru', 'tcn', or 'transformer'
        input_size: Number of input features
        forecast_horizon: Forecast steps
        **kwargs: Additional model-specific parameters
    
    Returns:
        PyTorch model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMForecaster(input_size, forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'gru':
        return GRUForecaster(input_size, forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'tcn':
        return TCNForecaster(input_size, forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'transformer':
        return TransformerForecaster(input_size, forecast_horizon=forecast_horizon, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'lstm', 'gru', 'tcn', 'transformer'")


if __name__ == '__main__':
    # Quick test
    print("Testing deep learning forecasters...")
    
    # Dummy data
    seq_len = 12
    n_features = 5
    forecast_horizon = 3
    
    X_dummy = np.random.randn(100, seq_len, n_features)
    y_dummy = np.random.randn(100, forecast_horizon)
    
    dataset = TimeSeriesDataset(X_dummy, y_dummy)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Test each model
    for model_name in ['lstm', 'gru', 'tcn', 'transformer']:
        print(f"\nTesting {model_name.upper()}...")
        model = get_model(model_name, input_size=n_features, forecast_horizon=forecast_horizon)
        
        # Forward pass
        X_batch, y_batch = next(iter(loader))
        output = model(X_batch)
        
        print(f"  Input shape: {X_batch.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ {model_name.upper()} working correctly")
    
    print("\n✓ All models initialized successfully!")
