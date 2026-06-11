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
from data_preprocessing import create_sequences


def run_evaluation():
    """Loads the test set and all trained models, generates predictions and calculates performance metrics, then creates visualizations."""

    print("Loading test data and models")
    test_data = pd.read_csv('data/processed/test_final.csv')
    
    # 2D Data for Baselines
    X_test_2d = test_data.drop(columns=['Appliances'])
    y_test_2d = test_data['Appliances']
    
    # 3D Data for Deep Learning
    SEQ_LENGTH = 24
    X_test_3d, y_test_3d = create_sequences(test_data, 'Appliances', seq_length=SEQ_LENGTH)
    
    # Load Models
    lr = joblib.load('models/linear_regression.joblib')
    rf = joblib.load('models/random_forest.joblib')
    xgb = joblib.load('models/xgboost.joblib')
    dl_model = tf.keras.models.load_model('models/cnn_lstm_tuned.keras')

    # Generate Predictions
    preds = {
        'Linear Regression': lr.predict(X_test_2d)[SEQ_LENGTH:],
        'Random Forest': rf.predict(X_test_2d)[SEQ_LENGTH:],
        'XGBoost': xgb.predict(X_test_2d)[SEQ_LENGTH:],
        'Tuned CNN-LSTM': dl_model.predict(X_test_3d).flatten()
    }
    
    # Align the true values to match the shortened sequence length
    y_true_aligned = y_test_3d

    results = {}
    for name, y_pred in preds.items():
        mae = mean_absolute_error(y_true_aligned, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred))
        results[name] = {'MAE': mae, 'RMSE': rmse, 'preds': y_pred}
        print(f"{name} - MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
    model_dir = 'models/images'
    os.makedirs(model_dir, exist_ok=True)
    generate_visualizations(results, y_true_aligned, model_dir)

def generate_visualizations(results, y_test, model_dir):
    """Generates and saves a suite of visualizations comparing model performance.
        Args:
            results (dict): Dictionary containing model names, their predictions, and performance metrics.
            y_test (np.array): The true target values for the test set.
            model_dir (str): Directory path to save the generated images.
    """
    print("\nGenerating Visualizations")
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