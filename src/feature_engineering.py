import pandas as pd
import numpy as np

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts cyclical time features from the datetime index."""
    df = df.copy()
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df.drop(columns=['hour', 'month', 'day_of_week'], inplace=True)
    return df

def create_rolling_and_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Short/Long term memory and autoregressive lags."""
    df = df.copy()
    
    # Rolling Windows (1-hour, 3-hour, and our custom 48-hour thermal inertia)
    # 10 min intervals -> 6=1hr, 18=3hr, 288=48hr
    for window, name in [(6, '1h'), (18, '3h'), (288, '48h')]:
        df[f'T_out_rolling_{name}'] = df['T_out'].rolling(window=window, min_periods=1).mean()
        df[f'T1_rolling_{name}'] = df['T1'].rolling(window=window, min_periods=1).mean()
        
    # Lagged Features (What happened 10 mins ago? 30 mins ago?)
    for lag, name in [(1, '10m'), (3, '30m')]:
        df[f'Appliances_lag_{name}'] = df['Appliances'].shift(lag)
        df[f'T_out_lag_{name}'] = df['T_out'].shift(lag)
        
    # Backfill the first few rows that get NaN from the shift
    df.bfill(inplace=True)
    
    return df

def create_domain_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interactions and Thermodynamic Physics."""
    df = df.copy()
    
    # Rubric Interactions (Basic Heat Index Proxies)
    df['Outdoor_Heat_Index'] = df['T_out'] * df['RH_out']
    df['Indoor_Comfort_Index'] = df['T1'] * df['RH_1']
    
    
    # Thermal Gradient (Newton's Law of Cooling Proxy)
    # Heat transfer rate is proportional to the difference between environments.
    df['Delta_T'] = df['T_out'] - df['T1']
    
    # Temperature Velocity (First Derivative)
    # HVAC systems don't just react to absolute temp; they react to how FAST it is dropping/rising.
    # diff() calculates the change over the last 10 minutes.
    df['T_out_velocity'] = df['T_out'].diff().fillna(0)
    df['T1_velocity'] = df['T1'].diff().fillna(0)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function to run the full engineering pipeline."""
    print("Engineering time-based features...")
    df = create_time_features(df)
    
    print("Engineering rolling windows and lagged features...")
    df = create_rolling_and_lagged_features(df)
    
    print("Engineering thermodynamic interactions...")
    df = create_domain_and_interaction_features(df)
    
    print(f"Feature engineering complete. New shape: {df.shape}")
    return df