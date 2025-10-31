import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from prepare_training_data import processed_data  # Import preprocessed data
import joblib

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
SEQUENCE_LENGTH = 70  # Number of past time steps to use for predictions
BATCH_SIZE = 64
EPOCHS = 100  # Increase for better accuracy
COINS = ["ETH", "SUI"]

# Create models directory
os.makedirs("models", exist_ok=True)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length, :-1])  # All indicator columns except price
        y.append(data[i + seq_length, -1])  # Next timestep price
    return np.array(X), np.array(y)

# Train model for each coin
for coin in COINS:
    print(f"Training model for {coin}...")

    df = processed_data[coin]

    # Select relevant features (indicators) and price
    features = ["EMA_10", "EMA_50", "MACD", "MACD_Signal", "RSI", "VWAP", "ATR",
                "Bollinger_Middle", "Bollinger_Upper", "Bollinger_Lower", "close"]
    df = df[features].dropna()

    # Normalize features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    joblib.dump(scaler, f"{coin}.pkl")

    # Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    # Split into training (80%) and testing (20%) sets
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Build LSTM model
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),  # return_sequences=False to output the last hidden state
        Dropout(0.2),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dense(1)  # Predict next price
    ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=MeanAbsoluteError(), metrics=["mse"])

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stopping])

    # Save trained model
    model.save(f"{coin}.keras")
