import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_baselines():
    """Trains baseline models and serializes them to disk."""
    print("Loading scaled data for baseline training")
    train_data = pd.read_csv('data/processed/train_final.csv')
    
    X_train = train_data.drop(columns=['Appliances'])
    y_train = train_data['Appliances']

    os.makedirs('models', exist_ok=True)

    print("Training Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'models/linear_regression.joblib')

    print("Tuning & Training Random Forest")
    rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 4]}
    rf = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=5, cv=3, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf.best_estimator_, 'models/random_forest.joblib')

    print("Tuning & Training XGBoost")
    xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    xgb = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_params, n_iter=5, cv=3, n_jobs=-1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb.best_estimator_, 'models/xgboost.joblib')

    print("Baselines trained and serialized successfully.")

if __name__ == "__main__":
    train_baselines()