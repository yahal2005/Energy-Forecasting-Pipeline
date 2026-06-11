import pandas as pd
import numpy as np
import os
import sys

# Suppress TensorFlow logging spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras_tuner as kt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.data_preprocessing import create_sequences

def load_and_sequence():
    """Loads the final processed datasets, applies the sequence generation for the LSTM, and returns the 3D arrays ready for tuning and evaluation."""
    print("Loading final datasets for Deep Learning Tuning...")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train_final.csv"))
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "val_final.csv"))
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "test_final.csv"))
    
    # Strips lingering strings just to be safe
    train = train.select_dtypes(include=['number'])
    val = val.select_dtypes(include=['number'])
    test = test.select_dtypes(include=['number'])

    SEQ_LENGTH = 24
    
    # Uses the sequence generaton
    X_train, y_train = create_sequences(train, config.TARGET, seq_length=SEQ_LENGTH)
    X_val, y_val = create_sequences(val, config.TARGET, seq_length=SEQ_LENGTH)
    X_test, y_test = create_sequences(test, config.TARGET, seq_length=SEQ_LENGTH)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, SEQ_LENGTH


def get_model_builder(seq_length, num_features):
    """
    Closure to pass the exact dynamic input shape into the KerasTuner builder.
    Args:
        seq_length (int): The length of the input sequences (e.g., 24).
        num_features (int): The number of features in the input data.
    Returns:
        function: A model-building function that KerasTuner can use.
    """
    def build_model(hp):
        """Definnes the CNN-LSTM architecture with hyperparameter to tune.
           Args:
               hp (HyperParameters): The KerasTuner object for defining hyperparameters.
            Returns:
               tf.keras.Model: The compiled Keras model.
        """
        model = Sequential()
        model.add(Input(shape=(seq_length, num_features)))
    
        hp_filters = hp.Int('conv_filters', min_value=32, max_value=128, step=32)
        model.add(Conv1D(filters=hp_filters, kernel_size=3, activation='relu')) 
        model.add(MaxPooling1D(pool_size=2)) # Compress the local noise before the LSTM
        
        # Tune the LSTM units
        hp_lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
        model.add(LSTM(hp_lstm_units, activation='relu', return_sequences=False))
        
        # Tune the Dropout rate
        hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)
        model.add(Dropout(hp_dropout))
        
        # Tune the Dense compression layer
        hp_dense = hp.Int('dense_units', min_value=16, max_value=64, step=16)
        model.add(Dense(hp_dense, activation='relu'))
        
        model.add(Dense(1, activation='linear'))
        
        # Tune the Learning Rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mae'])
        return model
    return build_model

def optimize_and_evaluate():
    """Runs the KerasTuner optimization process and evaluates the best model on the test set."""
    X_train, y_train, X_val, y_val, X_test, y_test, seq_length = load_and_sequence()
    num_features = X_train.shape[2]
    
    print(f"\nInitialized Sequences. Input Shape: {X_train.shape}")
    print("Initializing KerasTuner Random Search...")

    model_builder = get_model_builder(seq_length, num_features)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10, 
        executions_per_trial=1,
        directory=os.path.join(config.PROJECT_ROOT, 'models', 'tuner_logs'),
        project_name='energy_cnn_lstm',
        overwrite=True
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Commencing Hyperparameter Search...")
    tuner.search(
        X_train, y_train,
        epochs=30, 
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("Search Complete. Extracting the mathematically perfect architecture...")
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
    
    print("Training the Champion model fully...")
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
    
    print(f"\nPost-Optimization Performance on Test Set:")
    print(f"Tuned MAE:  {mae:.4f}")
    print(f"Tuned RMSE: {rmse:.4f}")
    
    os.makedirs(os.path.join(config.PROJECT_ROOT, "models"), exist_ok=True)
    best_model.save(os.path.join(config.PROJECT_ROOT, "models", "cnn_lstm_tuned.keras"))
    print("Champion model saved as 'cnn_lstm_tuned.keras'")

if __name__ == "__main__":
    optimize_and_evaluate()