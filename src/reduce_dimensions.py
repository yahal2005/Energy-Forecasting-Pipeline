import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def prune_features(df):
    # Features identified as dead weight from our Random Forest
    cols_to_drop = [
        'Date',
        'is_weekend', 'month_cos', 'month_sin', 
        'T_out_lag_30m', 'T_out_lag_10m',
        'rv1', 'rv2', 'lights'
    ]
    
    print(f"Dropping useless features to optimize for Deep Learning")
    
    df_reduced = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"New streamlined feature count: {df_reduced.shape[1] - 1} predictors.") 
    
    return df_reduced


if __name__ == "__main__":
    prune_features()