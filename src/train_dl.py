import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def load_and_reshape():
    print("Loading pristine final datasets...")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"), index_col=0)
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "val_final.csv"), index_col=0)
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "test_final.csv"), index_col=0)
    
    # Separate Features (X) and Target (y)
    X_train, y_train = train.drop(columns=[config.TARGET]).values, train[config.TARGET].values
    X_val, y_val = val.drop(columns=[config.TARGET]).values, val[config.TARGET].values
    X_test, y_test = test.drop(columns=[config.TARGET]).values, test[config.TARGET].values
    
    # Because we flattened our lags into columns, timesteps = 1
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"Data reshaped for CNN-LSTM. Input shape: {X_train.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn_lstm(input_shape):
    print("\nCompiling CNN-LSTM Architecture...")
    model = Sequential([
        # Spatial Extraction (CNN)
        Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape),
        
        # Temporal Sequence Learning (LSTM)
        LSTM(128, activation='relu', return_sequences=False),
        
        # Regularization (Prevent Overfitting)
        Dropout(0.2),
        
        # Deep Dense Layer for complex combinations
        Dense(32, activation='relu'),
        
        # Output Layer (1 continuous number for regression)
        Dense(1, activation='linear')
    ])
    
    # Adam Optimizer and MSE Loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_evaluate_dl():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_reshape()
    
    model = build_cnn_lstm((X_train.shape[1], X_train.shape[2]))
    
    # Early Stopping monitors the Validation Set. 
    # If the model stops improving for 10 epochs, it halts and restores the best weights.
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    print("Training")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("Deep Learning Performance")
    preds = model.predict(X_test, verbose=0)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Final Deep Learning Score      -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    
    # Save the architecture
    model.save(os.path.join(config.PROJECT_ROOT, "models", "cnn_lstm_model.keras"))
    print("Model architecture and weights saved to /models/")

if __name__ == "__main__":
    train_evaluate_dl()