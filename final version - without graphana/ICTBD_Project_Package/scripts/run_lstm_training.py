import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import warnings
import joblib

warnings.filterwarnings("ignore")
# %matplotlib inline # Not needed for script execution

# --- 1. Setup and Configuration ---
EPW_FILE = "/home/ubuntu/upload/Torino_IT-hour.epw"
SIM_FILE = "/home/ubuntu/upload/eplusout.csv"
os.makedirs("output_lstm", exist_ok=True)
N_LAG = 24
SPLIT_RATIO = 0.8
EPOCHS = 50
BATCH_SIZE = 64
QUANTILES = [0.1, 0.5, 0.9]

# --- 2. Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
try:
    epw = pd.read_csv(EPW_FILE, skiprows=8, header=None)
    epw.columns = ["Year", "Month", "Day", "Hour", "Minute", "DataSource", "DryBulb", "DewPoint", "RH", "AtmosPressure", 
                   "ExtGlobHorRad", "ExtDirNormRad", "ExtDifHorRad", "GlobalHorRad", "DirectNormRad", 
                   "DiffuseHorRad", "InfraSky", "WindDir", "WindSpd", "TotalSkyCover", "OpaqueSkyCover", 
                   "Visibility", "CeilingHeight", "PresWeatherObs", "PresWeatherCodes", "PrecipWater", "AerosolOptDepth",
                   "SnowDepth", "DaysSinceSnow", "Albedo", "LiquidPrecip", "RainRate", "RainDuration", "SnowRate", "SnowDuration"]
    weather_df = epw[["DryBulb", "RH", "ExtDirNormRad", "ExtDifHorRad"]].copy()
    weather_df.columns = ["Temperature", "Humidity", "DirectRad", "DiffuseRad"]
except FileNotFoundError:
    print(f"Error: EPW file not found at {EPW_FILE}")
    exit()

try:
    sim_df = pd.read_csv(SIM_FILE, low_memory=False)
    target_col = "BLOCCO1:ZONA3:Zone Total Internal Latent Gain Energy [J](TimeStep)"
    if target_col not in sim_df.columns:
        print(f"Error: Target column \t{target_col}\t not found in {SIM_FILE}")
        energy_cols = [c for c in sim_df.columns if "Energy" in c and "Latent" in c]
        if energy_cols:
            target_col = energy_cols[0]
            print(f"Using fallback target column: {target_col}")
        else:
            print("No suitable energy column found. Exiting.")
            exit()
    sim_df_hourly = sim_df[[target_col]].copy()
except FileNotFoundError:
    print(f"Error: Simulation file not found at {SIM_FILE}")
    exit()

weather_df_reset = weather_df.reset_index(drop=True)
sim_df_hourly_reset = sim_df_hourly.reset_index(drop=True)
min_length = min(len(weather_df_reset), len(sim_df_hourly_reset))
weather_df_reset = weather_df_reset.iloc[:min_length]
sim_df_hourly_reset = sim_df_hourly_reset.iloc[:min_length]
full_df = pd.concat([weather_df_reset, sim_df_hourly_reset], axis=1).dropna()
print(f"Combined dataset shape: {full_df.shape}")

# --- 3. Feature Scaling and Sequence Creation ---
features = full_df[["Temperature", "Humidity", "DirectRad", "DiffuseRad"]].values
target = full_df[target_col].values
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
Y_scaled = scaler_Y.fit_transform(target.reshape(-1, 1))
X_seq, Y_seq = [], []
for i in range(N_LAG, len(X_scaled)):
    X_seq.append(X_scaled[i - N_LAG:i])
    Y_seq.append(Y_scaled[i])
X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)
split_idx = int(SPLIT_RATIO * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
Y_train, Y_test = Y_seq[:split_idx], Y_seq[split_idx:]
print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, Y={Y_test.shape}")

# --- 4. LSTM Model with Quantile Loss ---
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return keras.backend.mean(keras.backend.maximum(q * e, (q - 1) * e), axis=-1)

def build_lstm_quantile_model(n_timesteps, n_features, n_quantiles):
    model = keras.Sequential()
    model.add(layers.LSTM(50, activation="relu", input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(layers.LSTM(50, activation="relu"))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(n_quantiles))
    return model

n_features = X_train.shape[2]
n_quantiles = len(QUANTILES)
lstm_model = build_lstm_quantile_model(N_LAG, n_features, n_quantiles)
losses = [lambda y_true, y_pred, q=q: quantile_loss(q, y_true, y_pred) for q in QUANTILES]
lstm_model.compile(optimizer="adam", loss=losses)
lstm_model.summary()

# --- 5. Train the LSTM Model ---
print("Training LSTM model...")
history = lstm_model.fit(
    X_train,
    Y_train, 
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
)
print("Training complete.")
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Model Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("output_lstm/training_history.png")
# plt.show() # Avoid showing plot

# --- 6. Evaluate the Model and Visualize Predictions ---
print("Evaluating model on test data...")
Y_pred_quantiles_scaled = lstm_model.predict(X_test)
Y_pred_quantiles = scaler_Y.inverse_transform(Y_pred_quantiles_scaled)
Y_test_actual = scaler_Y.inverse_transform(Y_test)
Y_pred_lower = Y_pred_quantiles[:, 0]
Y_pred_median = Y_pred_quantiles[:, 1]
Y_pred_upper = Y_pred_quantiles[:, 2]
mae = mean_absolute_error(Y_test_actual, Y_pred_median)
rmse = np.sqrt(mean_squared_error(Y_test_actual, Y_pred_median))
print(f"Test MAE (Median Prediction): {mae:.2f}")
print(f"Test RMSE (Median Prediction): {rmse:.2f}")

plt.figure(figsize=(15, 7))
plt.plot(Y_test_actual, label="True Energy", color="black")
plt.plot(Y_pred_median, label="Predicted Median (q=0.5)", color="purple")
plt.fill_between(range(len(Y_test_actual)), Y_pred_lower, Y_pred_upper, color="purple", alpha=0.2, label="80% Prediction Interval (q=0.1 to 0.9)")
plt.title("LSTM Energy Prediction with Uncertainty (Quantile Regression)")
plt.xlabel("Time Step (Hour)")
plt.ylabel("Energy (J)")
plt.legend()
plt.grid(True)
plt.savefig("output_lstm/prediction_vs_actual_quantile.png")
# plt.show() # Avoid showing plot

plt.figure(figsize=(15, 7))
plt.plot(Y_test_actual[:100], label="True Energy", color="black")
plt.plot(Y_pred_median[:100], label="Predicted Median (q=0.5)", color="purple")
plt.fill_between(range(100), Y_pred_lower[:100], Y_pred_upper[:100], color="purple", alpha=0.2, label="80% Prediction Interval (q=0.1 to 0.9)")
plt.title("LSTM Energy Prediction - First 100 Hours")
plt.xlabel("Time Step (Hour)")
plt.ylabel("Energy (J)")
plt.legend()
plt.grid(True)
plt.savefig("output_lstm/prediction_vs_actual_quantile_100h.png")
# plt.show() # Avoid showing plot

# --- 7. Save Model and Scalers ---
lstm_model.save("/home/ubuntu/lstm_energy_predictor.keras")
joblib.dump(scaler_X, "/home/ubuntu/scaler_X.joblib")
joblib.dump(scaler_Y, "/home/ubuntu/scaler_Y.joblib")
print("Model and scalers saved.")

