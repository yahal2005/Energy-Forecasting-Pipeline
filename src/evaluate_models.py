import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def load_test_data():
    print("Loading test data for evaluation")
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "test_final.csv"), index_col=0)
    X_test = test.drop(columns=[config.TARGET]).values
    y_test = test[config.TARGET].values
    return X_test, y_test

def run_evaluation():
    X_test, y_test = load_test_data()
    model_dir = os.path.join(config.PROJECT_ROOT, "models")
    
    print("Loading Serialized Models")
    lr = joblib.load(os.path.join(model_dir, "linear_regression.joblib"))
    rf = joblib.load(os.path.join(model_dir, "random_forest.joblib"))
    xgb = joblib.load(os.path.join(model_dir, "xgboost.joblib"))
    
    X_test_dl = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    dl_model = tf.keras.models.load_model(os.path.join(model_dir, "cnn_lstm_tuned.keras"))
    
    print("Executing Predictions")
    results = {
        "Linear Regression": {"preds": lr.predict(X_test)},
        "Random Forest": {"preds": rf.predict(X_test)},
        "XGBoost": {"preds": xgb.predict(X_test)},
        "Tuned CNN-LSTM": {"preds": dl_model.predict(X_test_dl, verbose=0).flatten()}
    }
    
    print("METRIC Comparison")
    for name, data in results.items():
        data["MAE"] = mean_absolute_error(y_test, data["preds"])
        data["RMSE"] = np.sqrt(mean_squared_error(y_test, data["preds"]))
        print(f"{name.ljust(18)} -> MAE: {data['MAE']:.4f} | RMSE: {data['RMSE']:.4f}")
        
    generate_visualizations(results, y_test, model_dir)

def generate_visualizations(results, y_test, model_dir):
    print("\nGenerating 5 World-Class Visualizations for Subtask 4.4...")
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Performance Bar Chart
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    names = list(results.keys())
    maes = [results[n]["MAE"] for n in names]
    rmses = [results[n]["RMSE"] for n in names]
    
    sns.barplot(x=names, y=maes, ax=ax[0], hue=names, palette="mako", legend=False)
    ax[0].set_title("Mean Absolute Error", fontweight='bold')
    ax[0].set_ylabel("MAE")
    ax[0].tick_params(axis='x', rotation=15)
    
    sns.barplot(x=names, y=rmses, ax=ax[1], hue=names, palette="flare", legend=False)
    ax[1].set_title("Root Mean Squared Error", fontweight='bold')
    ax[1].set_ylabel("RMSE")
    ax[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "1_metric_comparison.png"))
    
    # PLOT 2: Predicted vs Actual 
    slice_idx = 200
    plt.figure(figsize=(16, 6))
    plt.plot(y_test[:slice_idx], label="Actual Energy Use", color="black", linewidth=2.5)
    plt.plot(results["Linear Regression"]["preds"][:slice_idx], label="Linear Reg", color="orange", alpha=0.6, linestyle=":")
    plt.plot(results["XGBoost"]["preds"][:slice_idx], label="XGBoost", color="blue", alpha=0.6, linestyle="-.")
    plt.plot(results["Tuned CNN-LSTM"]["preds"][:slice_idx], label="CNN-LSTM", color="red", linewidth=2)
    plt.title("Actual vs. Predicted Energy Consumption", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "2_timeseries_predictions.png"))
    
    # PLOT 3: 45-Degree Scatter Plot
    plt.figure(figsize=(8, 8))
    champ_preds = results["Tuned CNN-LSTM"]["preds"]
    plt.scatter(y_test, champ_preds, alpha=0.3, color="teal")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color="red")
    plt.title("CNN-LSTM: Actual vs. Predicted (45-Degree Reference)", fontweight='bold')
    plt.xlabel("Actual Scaled Consumption")
    plt.ylabel("Predicted Scaled Consumption")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "3_actual_vs_predicted_scatter.png"))

    # PLOT 4: Residual Error Scatter Plot 
    plt.figure(figsize=(10, 6))
    residuals = y_test - champ_preds
    sns.scatterplot(x=y_test, y=residuals, color="purple", alpha=0.4)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Residual Plot: Where is the model making mistakes?", fontweight='bold')
    plt.xlabel("Actual Energy Consumption")
    plt.ylabel("Prediction Error (Residual)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "4_residual_scatter.png"))

    # PLOT 5: Residual Error Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True, color="indigo")
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Error Distribution", fontweight='bold')
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "5_residual_distribution.png"))
    
    print("Evaluation complete and all 5 charts saved!")

if __name__ == "__main__":
    run_evaluation()