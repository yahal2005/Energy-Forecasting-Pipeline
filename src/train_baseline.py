import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def load_train_data():
    print("Loading training data...")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"), index_col=0)
    X_train = train.drop(columns=[config.TARGET]).values
    y_train = train[config.TARGET].values
    return X_train, y_train


def train_and_save_baselines():
    X_train, y_train = load_train_data()
    model_dir = os.path.join(config.PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(model_dir, "linear_regression.joblib"))

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(model_dir, "random_forest.joblib"))

    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(model_dir, "xgboost.joblib"))

    print("All baseline models trained and serialized to /models/")

if __name__ == "__main__":
    train_and_save_baselines()