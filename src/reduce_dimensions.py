import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def prune_features():
    print("Loading processed splits")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train.csv"), index_col=0)
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "val.csv"), index_col=0)
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "test.csv"), index_col=0)

    # Features identified as dead weight from our Random Forest
    cols_to_drop = [
        'is_weekend', 'month_cos', 'month_sin', 
        'T_out_lag_30m', 'T_out_lag_10m',
        'rv1', 'rv2', 'lights' # Cleaning up the random variables as well
    ]
    
    print(f"Dropping useless features to optimize for Deep Learning")
    
    
    train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    val.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    print(f"New streamlined feature count: {train.shape[1] - 1} predictors.") 
    
    # Save them as the final datasets
    train.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"))
    val.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "val_final.csv"))
    test.to_csv(os.path.join(config.PROCESSED_DATA_PATH, "test_final.csv"))
    print("Pruned datasets saved successfully!")

if __name__ == "__main__":
    prune_features()