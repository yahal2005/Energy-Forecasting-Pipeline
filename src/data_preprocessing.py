import sys

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def split_time_series(df, train_ratio=0.7, val_ratio=0.2):
    """Strict chronological split (70/20/10) to prevent data leakage."""
    train_idx = int(len(df) * train_ratio)
    val_idx = int(len(df) * (train_ratio + val_ratio))
    
    train = df.iloc[:train_idx]
    val = df.iloc[train_idx:val_idx]
    test = df.iloc[val_idx:]
    
    return train, val, test

def scale_data(train_df, val_df, test_df):
    """Scales data using RobustScaler to handle massive energy spikes safely."""
    scaler = RobustScaler()
    
    # Fit strictly on train to prevent leakage
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    val_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    
    return train_scaled, val_scaled, test_scaled, scaler

def create_sequences(data_df, target_col='energy_consumed', seq_length=24):
    """
    Transforms 2D tabular data into 3D sequential blocks for the LSTM.
    Shape goes from (samples, features) -> (samples, seq_length, features)
    """
    target_idx = data_df.columns.get_loc(target_col)
    data_values = data_df.values
    
    X, y = [], []
    for i in range(len(data_values) - seq_length):
        X.append(data_values[i:(i + seq_length)]) # The 24-step lookback window
        y.append(data_values[i + seq_length, target_idx]) # The target to predict
        
    return np.array(X), np.array(y)

def save_processed_data(train_scaled: pd.DataFrame, val_scaled: pd.DataFrame, test_scaled: pd.DataFrame, scaler):
    """
    Saves the final scaled datasets and the fitted scaler object for model training.
    """
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    
    # Save the dataframes as CSVs
    train_scaled.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"))
    val_scaled.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "val_final.csv"))
    test_scaled.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "test_final.csv"))
    
    # Save the scaler object so we can un-scale predictions later
    joblib.dump(scaler, os.path.join(config.PROCESSED_DATA_PATH, "scaler.save"))
    
    print(f"Success! Processed data and scaler saved to {config.PROCESSED_DATA_PATH}")