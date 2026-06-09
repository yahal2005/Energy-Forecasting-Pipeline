import pandas as pd
import numpy as np
import os
import sys

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras_tuner as kt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def load_and_reshape():
    print("Loading final datasets for tuning")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"), index_col=0)
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "val_final.csv"), index_col=0)
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "test_final.csv"), index_col=0)
    
    X_train, y_train = train.drop(columns=[config.TARGET]).values, train[config.TARGET].values
    X_val, y_val = val.drop(columns=[config.TARGET]).values, val[config.TARGET].values
    X_test, y_test = test.drop(columns=[config.TARGET]).values, test[config.TARGET].values
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_model_builder(input_shape):
    """
    Closure to pass the exact dynamic input shape into the KerasTuner builder.
    """
    def build_model(hp):
        model = Sequential()
        
        # Modern Keras 3 syntax: Use Input layer explicitly
        model.add(Input(shape=input_shape))
        
        # Tune the number of CNN filters
        hp_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
        model.add(Conv1D(filters=hp_filters, kernel_size=1, activation='relu')) 
        
        # Tune the LSTM units
        hp_lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
        model.add(LSTM(hp_lstm_units, activation='relu', return_sequences=False))
        
        # Subtask 5.2: Tune the Dropout rate
        hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)
        model.add(Dropout(hp_dropout))
        
        # Tune the Dense compression layer
        hp_dense = hp.Int('dense_units', min_value=16, max_value=64, step=16)
        model.add(Dense(hp_dense, activation='relu'))
        
        model.add(Dense(1, activation='linear'))
        
        # Tune the Learning Rate for the Adam optimizer
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mae'])
        return model
    return build_model

def optimize_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_reshape()
    
    print("\nInitializing KerasTuner Random Search...")

    model_builder = get_model_builder((1, X_train.shape[2]))

    # Using RandomSearch to test 10 random combinations of the architecture.
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10, 
        executions_per_trial=1,
        directory=os.path.join(config.PROJECT_ROOT, 'models', 'tuner_logs'),
        project_name='energy_cnn_lstm',
        overwrite=True
    )
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Commencing Hyperparameter Search")
    tuner.search(
        X_train, y_train,
        epochs=30, 
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Model Evaluation Post-Optimization
    print("Search Complete. Extracting the mathematically perfect architecture")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    --- OPTIMAL HYPERPARAMETERS FOUND ---
    CNN Filters:   {best_hps.get('conv_filters')}
    LSTM Units:    {best_hps.get('lstm_units')}
    Dropout Rate:  {best_hps.get('dropout_rate')}
    Dense Units:   {best_hps.get('dense_units')}
    Learning Rate: {best_hps.get('learning_rate')}
    """)
    
    best_model = tuner.hypermodel.build(best_hps)
    
    print("Training the Champion model fully")
    best_model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    preds = best_model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Post-Optimization Performance")
    print(f"Pre-Tuning MAE:  0.0249  | Tuned MAE:  {mae:.4f}")
    print(f"Pre-Tuning RMSE: 0.0590  | Tuned RMSE: {rmse:.4f}")
    
    # Save the best model 
    best_model.save(os.path.join(config.PROJECT_ROOT, "models", "cnn_lstm_tuned.keras"))
    print("Champion model saved as 'cnn_lstm_tuned.keras'")

if __name__ == "__main__":
    optimize_and_evaluate()