import pandas as pd
import numpy as np
import os
import sys

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import create_sequences
from src import config

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def train_dl():
    """Trains a deep learning model for energy forecasting."""
    print("Loading final datasets for Deep Learning")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"))
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "val_final.csv"))

    # Purge lingering strings
    train = train.select_dtypes(include=['number'])
    val = val.select_dtypes(include=['number'])

    SEQ_LENGTH = 24
    
    X_train, y_train = create_sequences(train, config.TARGET, seq_length=SEQ_LENGTH)
    X_val, y_val = create_sequences(val, config.TARGET, seq_length=SEQ_LENGTH)
    
    num_features = X_train.shape[2]
    print(f"DL Input Shape: {X_train.shape} -> (samples, sequences, features)")

    # Build the baseline CNN-LSTM
    model = Sequential([
        Input(shape=(SEQ_LENGTH, num_features)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training baseline CNN-LSTM...")
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    os.makedirs(os.path.join(config.PROJECT_ROOT, "models"), exist_ok=True)
    model.save(os.path.join(config.PROJECT_ROOT, "models", "cnn_lstm_baseline.keras"))
    print("Baseline DL model saved as 'cnn_lstm_baseline.keras'")

if __name__ == "__main__":
    train_dl()