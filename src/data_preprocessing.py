import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from src import config

def split_time_series(df: pd.DataFrame):
    """
    Data Splitting.
    Implements a strict sequential 70/20/10 (Train/Val/Test) split
    to prevent temporal data leakage.
    """
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9) 
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Temporal Split Complete (70/20/10):")
    print(f"Train set: {len(train_df)} rows")
    print(f"Val set:   {len(val_df)} rows")
    print(f"Test set:  {len(test_df)} rows")
    
    return train_df, val_df, test_df

def scale_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Data Scaling.
    Scaler is fitted only on the 70% Training set to prevent future leakage.
    """
    scaler = MinMaxScaler()
    columns = train_df.columns
    
    # Fit strictly on train, transform all three splits
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=columns, index=train_df.index)
    val_scaled = pd.DataFrame(scaler.transform(val_df), columns=columns, index=val_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
    
    print("Scaling Complete: Applied Min-Max Scaling based on Train distribution.")
    return train_scaled, val_scaled, test_scaled, scaler

def save_processed_data(train_scaled: pd.DataFrame, val_scaled: pd.DataFrame, test_scaled: pd.DataFrame, scaler):
    """
    Saves the final scaled datasets and the fitted scaler object for model training.
    """
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    
    # Save the dataframes as CSVs
    train_scaled.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "train.csv"))
    val_scaled.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "val.csv"))
    test_scaled.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "test.csv"))
    
    # Save the scaler object so we can un-scale predictions later
    joblib.dump(scaler, os.path.join(config.PROCESSED_DATA_PATH, "scaler.save"))
    
    print(f"Success! Processed data and scaler saved to {config.PROCESSED_DATA_PATH}")