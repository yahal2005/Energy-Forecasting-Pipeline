import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def load_splits():
    print("Loading processed data...")

    # Load the preprocessed datasets (already split and scaled)
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train.csv"), index_col=0)
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "val.csv"), index_col=0)
    
    # Separate Features (X) and Target (y)
    X_train = train.drop(columns=[config.TARGET])
    y_train = train[config.TARGET]
    
    X_val = val.drop(columns=[config.TARGET])
    y_val = val[config.TARGET]
    
    return X_train, y_train, X_val, y_val

def train_evaluate_rf(X_train, y_train, X_val, y_val):
    print("Training Random Forest Baseline")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Evaluating on Validation Set")
    preds = rf.predict(X_val)
    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    print(f"\n--- Baseline Performance (Scaled Space) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return rf

def plot_feature_importances(model, feature_names):
    print("\nExtracting Feature Importances")
    importances = model.feature_importances_
    
    # Create a DataFrame to sort the math
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Generate the visual proof for your report
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='teal')
    plt.title("Random Forest Feature Importance (Algorithmic Selection)")
    plt.xlabel("Relative Predictive Power")
    plt.tight_layout()
    plt.show()
    
    print("\nBottom 5 Features (Candidates for Deletion):")
    print(fi_df.head(5).to_string(index=False))

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_splits()
    rf_model = train_evaluate_rf(X_train, y_train, X_val, y_val)
    plot_feature_importances(rf_model, X_train.columns)