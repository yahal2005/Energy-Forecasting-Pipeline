import pandas as pd
import data.config as config

def load_raw_data(filepath: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw energy dataset and enforces the chronological index.
    """
    print(f"Loading data from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data at {filepath}. Check your paths!")

    # Parse datetime
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    
    # Set as index
    df.set_index(config.DATETIME_COL, inplace=True)
    
    # Sort chronologically
    df.sort_index(inplace=True)

    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"ime range: {df.index.min()} to {df.index.max()}")
    
    return df