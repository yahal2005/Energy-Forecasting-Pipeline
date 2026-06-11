
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import create_sequences

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras import layers

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
    
    # timesteps = 1
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"Data reshaped for CNN-LSTM. Input shape: {X_train.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn_lstm(seq_length, num_features):
    inputs = keras.Input(shape=(seq_length, num_features))
    
    # CNN Layer to filter local noise across the 24-step window
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM Layer with actual sequence memory
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_dl():
    train_data = pd.read_csv('data/processed/train_final.csv')
    val_data = pd.read_csv('data/processed/val_final.csv')

    print("Generating 24-step sequences for Deep Learning")
    SEQ_LENGTH = 24
    X_train, y_train = create_sequences(train_data, 'Appliances', seq_length=SEQ_LENGTH)
    X_val, y_val = create_sequences(val_data, 'Appliances', seq_length=SEQ_LENGTH)

    num_features = X_train.shape[2]
    
    print(f"DL Input Shape: {X_train.shape} -> (samples, sequences, features)")

    model = build_cnn_lstm(SEQ_LENGTH, num_features)
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training CNN-LSTM")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stopping]
    )
    
    model.save('models/cnn_lstm_tuned.keras')
    print("Deep Learning model saved.")

if __name__ == "__main__":
    train_dl()