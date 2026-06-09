import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def run_feature_selection():
    print("Loading pre-selection data")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train.csv"), index_col=0)
    
    X_train = train.drop(columns=[config.TARGET])
    y_train = train[config.TARGET]
    
    print("Training Diagnostic Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("\nExtracting Feature Importances for Subtask 2.6...")
    importances = rf.feature_importances_
    
    fi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Generate the visual proof for your report
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='teal')
    plt.title("Random Forest Feature Importance (Algorithmic Selection)")
    plt.xlabel("Relative Predictive Power")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PROJECT_ROOT, "models", "feature_importance.png"))
    print("Feature Importance chart saved to /models/feature_importance.png")
    
    print("\nBottom 5 Features (The ones we dropped):")
    print(fi_df.head(5).to_string(index=False))

if __name__ == "__main__":
    run_feature_selection()