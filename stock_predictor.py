# stock_predictor.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
import os

# Suppress TensorFlow logging (optional, keeps output cleaner)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = "2023-11-13"
SEQUENCE_LENGTH = 60
TRAIN_SPLIT_RATIO = 0.8
EPOCHS = 100 # As specified in the examples
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LSTM_UNITS = 50
GRU_UNITS = 50
DENSE_UNITS = 25

# --- Helper Function for Sequence Creation ---
# Handles both single and multiple features
def create_sequences(data, seq_length, target_col_index=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :]) # Get all features for the sequence
        y.append(data[i + seq_length, target_col_index]) # Target is a specific column's value
    return np.array(X), np.array(y)

# --- Fetch Data Once ---
print(f"Fetching data for {TICKER}...")
stock_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
if stock_data.empty:
    print(f"Error: No data fetched for {TICKER}. Exiting.")
    exit()
print("Data fetched.")

# =============================================================================
# Model 1: LSTM (Close Price Only)
# =============================================================================
print("\n--- Starting LSTM Model (Close Price Only) ---")

# Prepare data
df_lstm = stock_data[['Close']].copy() # Use copy to avoid SettingWithCopyWarning
scaler_lstm = MinMaxScaler()
scaled_data_lstm = scaler_lstm.fit_transform(df_lstm['Close'].values.reshape(-1, 1))

# Create sequences (using helper function, data shape is (n_samples, 1))
X_lstm, y_lstm = create_sequences(scaled_data_lstm, SEQUENCE_LENGTH, target_col_index=0)

# Split data
train_size_lstm = int(len(X_lstm) * TRAIN_SPLIT_RATIO)
X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

# Reshape input for LSTM [samples, time steps, features]
# Already handled by create_sequences if input data has 1 column,
# but we ensure it's explicitly (samples, seq_length, 1)
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

# Build LSTM model
model_lstm = Sequential([
    LSTM(LSTM_UNITS, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    LSTM(LSTM_UNITS, return_sequences=False),
    Dense(DENSE_UNITS),
    Dense(1)
])

# Compile and Train
model_lstm.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
print("Training LSTM model...")
model_lstm.fit(X_train_lstm, y_train_lstm, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)

# Make predictions
print("Predicting with LSTM model...")
predictions_lstm_scaled = model_lstm.predict(X_test_lstm)
predictions_lstm = scaler_lstm.inverse_transform(predictions_lstm_scaled)

# Inverse transform actual values
y_test_lstm_unscaled = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1))

# Calculate RMSE
rmse_lstm = np.sqrt(np.mean((predictions_lstm - y_test_lstm_unscaled)**2))
print(f"LSTM Model (Close Only) RMSE: {rmse_lstm:.4f}")
print("--- Finished LSTM Model ---")


# =============================================================================
# Model 2: GRU (Close Price Only)
# =============================================================================
print("\n--- Starting GRU Model (Close Price Only) ---")

# Prepare data (Similar to LSTM, using Close price only)
df_gru_close = stock_data[['Close']].copy()
scaler_gru_close = MinMaxScaler()
scaled_data_gru_close = scaler_gru_close.fit_transform(df_gru_close['Close'].values.reshape(-1, 1))

# Create sequences
X_gru_close, y_gru_close = create_sequences(scaled_data_gru_close, SEQUENCE_LENGTH, target_col_index=0)

# Split data
train_size_gru_close = int(len(X_gru_close) * TRAIN_SPLIT_RATIO)
X_train_gru_close, X_test_gru_close = X_gru_close[:train_size_gru_close], X_gru_close[train_size_gru_close:]
y_train_gru_close, y_test_gru_close = y_gru_close[:train_size_gru_close], y_gru_close[train_size_gru_close:]

# Reshape input for GRU [samples, time steps, features]
X_train_gru_close = X_train_gru_close.reshape(X_train_gru_close.shape[0], X_train_gru_close.shape[1], 1)
X_test_gru_close = X_test_gru_close.reshape(X_test_gru_close.shape[0], X_test_gru_close.shape[1], 1)

# Build GRU model
model_gru_close = Sequential([
    GRU(GRU_UNITS, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    GRU(GRU_UNITS, return_sequences=False),
    Dense(DENSE_UNITS),
    Dense(1)
])

# Compile and Train
model_gru_close.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
print("Training GRU model (Close Only)...")
model_gru_close.fit(X_train_gru_close, y_train_gru_close, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)

# Make predictions
print("Predicting with GRU model (Close Only)...")
predictions_gru_close_scaled = model_gru_close.predict(X_test_gru_close)
predictions_gru_close = scaler_gru_close.inverse_transform(predictions_gru_close_scaled)

# Inverse transform actual values
y_test_gru_close_unscaled = scaler_gru_close.inverse_transform(y_test_gru_close.reshape(-1, 1))

# Calculate RMSE
rmse_gru_close = np.sqrt(np.mean((predictions_gru_close - y_test_gru_close_unscaled)**2))
print(f"GRU Model (Close Only) RMSE: {rmse_gru_close:.4f}")
print("--- Finished GRU Model (Close Only) ---")


# =============================================================================
# Model 3: GRU (Close Price + Volume)
# =============================================================================
print("\n--- Starting GRU Model (Close Price + Volume) ---")

# Prepare data
features_gru_vol = ['Close', 'Volume']
df_gru_vol = stock_data[features_gru_vol].copy()
scaler_gru_vol = MinMaxScaler() # Scale Close and Volume together
scaled_data_gru_vol = scaler_gru_vol.fit_transform(df_gru_vol)

# Create sequences (Target is still 'Close', which is index 0)
# X will have shape (samples, seq_length, 2)
X_gru_vol, y_gru_vol = create_sequences(scaled_data_gru_vol, SEQUENCE_LENGTH, target_col_index=0)

# Split data
train_size_gru_vol = int(len(X_gru_vol) * TRAIN_SPLIT_RATIO)
X_train_gru_vol, X_test_gru_vol = X_gru_vol[:train_size_gru_vol], X_gru_vol[train_size_gru_vol:]
y_train_gru_vol, y_test_gru_vol = y_gru_vol[:train_size_gru_vol], y_gru_vol[train_size_gru_vol:]

# Input shape is now (SEQUENCE_LENGTH, 2 features)
input_shape_gru_vol = (SEQUENCE_LENGTH, len(features_gru_vol))

# Build GRU model
model_gru_vol = Sequential([
    GRU(GRU_UNITS, return_sequences=True, input_shape=input_shape_gru_vol),
    GRU(GRU_UNITS, return_sequences=False),
    Dense(DENSE_UNITS),
    Dense(1) # Output layer predicts one value (the Close price)
])

# Compile and Train
model_gru_vol.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
print("Training GRU model (Close + Volume)...")
model_gru_vol.fit(X_train_gru_vol, y_train_gru_vol, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)

# Make predictions
print("Predicting with GRU model (Close + Volume)...")
predictions_gru_vol_scaled = model_gru_vol.predict(X_test_gru_vol)

# Inverse transform predictions
# We need to reconstruct the 2-feature structure to use the scaler properly
# Create dummy array of zeros with the same shape as predictions for the 'Volume' column
dummy_volume_pred = np.zeros_like(predictions_gru_vol_scaled)
# Concatenate predicted 'Close' (col 0) and dummy 'Volume' (col 1)
predictions_combined = np.concatenate((predictions_gru_vol_scaled, dummy_volume_pred), axis=1)
# Inverse transform using the scaler fitted on ['Close', 'Volume']
predictions_gru_vol = scaler_gru_vol.inverse_transform(predictions_combined)[:, 0] # Take only the first column (Close price)

# Inverse transform actual values (y_test)
# We need the corresponding scaled 'Volume' data from the test set to inverse transform y_test correctly
# Get the scaled 'Volume' part from the original scaled data corresponding to the y_test timeframe
start_index_y_test = train_size_gru_vol + SEQUENCE_LENGTH
end_index_y_test = start_index_y_test + len(y_test_gru_vol)
actual_volume_scaled_test = scaled_data_gru_vol[start_index_y_test:end_index_y_test, 1] # Index 1 is Volume
# Combine actual scaled 'Close' (y_test_gru_vol) and actual scaled 'Volume'
y_test_combined = np.concatenate((y_test_gru_vol.reshape(-1, 1), actual_volume_scaled_test.reshape(-1, 1)), axis=1)
y_test_gru_vol_unscaled = scaler_gru_vol.inverse_transform(y_test_combined)[:, 0] # Take only the first column (Close price)

# Calculate RMSE
rmse_gru_vol = np.sqrt(np.mean((predictions_gru_vol - y_test_gru_vol_unscaled)**2))
print(f"GRU Model (Close + Volume) RMSE: {rmse_gru_vol:.4f}")
print("--- Finished GRU Model (Close + Volume) ---")

print("\nAll models completed.")