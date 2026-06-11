import sys
import os
import pandas as pd

# Suppress TF logs if necessary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features
from src.reduce_dimensions import prune_features 
from src.data_preprocessing import split_time_series, scale_data, save_processed_data

def execute_data_pipeline():
    """Executes the entire data processing pipeline."""

    print("Loading Raw Data")
    df = load_raw_data()
    
    print("Engineering Features")
    df_engineered = engineer_features(df)
    
    print("Reducing Dimensions ")
    df_reduced = prune_features(df_engineered) 
    
    print("Temporal Split (70/20/10)")
    train, val, test = split_time_series(df_reduced)
    
    print("Scaling Data")
    train_scaled, val_scaled, test_scaled, scaler = scale_data(train, val, test)
    
    print("Saving Final Datasets")
    save_processed_data(train_scaled, val_scaled, test_scaled, scaler)
    print("Pipeline execution complete. Ready for ML Training.")

if __name__ == "__main__":
    execute_data_pipeline()