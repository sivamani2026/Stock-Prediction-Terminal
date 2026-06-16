import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing tensorflow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow/Keras is not installed. LSTM model will not be available.")

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Trains a Linear Regression model.
    """
    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression training complete.")
    return model

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor model.
    """
    logger.info("Training Random Forest Regressor model...")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return model

def build_lstm_model(input_shape: tuple) -> Any:
    """
    Builds a Sequential LSTM model with keras.
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow/Keras is not available. Cannot build LSTM model.")
        
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 15, batch_size: int = 32) -> Any:
    """
    Trains the LSTM model.
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow/Keras is not available. Cannot train LSTM model.")
        
    logger.info("Building LSTM model...")
    input_shape = (X_train.shape[1], 1)
    model = build_lstm_model(input_shape)
    
    logger.info(f"Training LSTM model (epochs={epochs}, batch_size={batch_size})...")
    # Using validation split to monitor overfitting
    model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1, 
        verbose=0
    )
    logger.info("LSTM training complete.")
    return model
