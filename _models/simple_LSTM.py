import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

######################################################
# Setup
######################################################
log_dir = "_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "simple_LSTM.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

model_dir = "_models"
os.makedirs(model_dir, exist_ok=True)

plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

######################################################
# Parameters
######################################################
SEQ_LENGTH = 20
DT = 0.1  # time step for finite difference
FEATURE_COLUMNS = ['time', 'position', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]
TARGET_COLUMN = 'velocity'

######################################################
# Data
######################################################
data_path = os.path.join("data", "cleaned_BUS.csv")
df = pd.read_csv(data_path)
logging.info(f"Loaded training data from {data_path} with shape {df.shape}.")

# Normalize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[FEATURE_COLUMNS + [TARGET_COLUMN]])
logging.info("Normalized input data.")

# Create LSTM sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 2])  # index 2 = velocity
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH)
logging.info(f"Created sequences with shape: X={X.shape}, y={y.shape}.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Split data into training and testing sets.")

######################################################
# Model
######################################################
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
logging.info("Compiled LSTM model.")

history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
logging.info("Trained LSTM model.")

y_pred = model.predict(X_test).flatten()
mse = np.mean((y_test - y_pred) ** 2)
logging.info(f"Model evaluation — MSE: {mse:.4f}")

######################################################
# Save Results
######################################################
results_path = os.path.join(results_dir, "simple_LSTM.txt")
with open(results_path, "w") as f:
    f.write(f"Mean Squared Error: {mse:.4f}\n")
logging.info(f"Saved model evaluation to {results_path}.")

model_path = os.path.join(model_dir, "train_lstm_model.h5")
model.save(model_path)
logging.info(f"Saved trained model to {model_path}.")

######################################################
# Plot: Velocity + Derived Acceleration and Jerk
######################################################
# Reconstruct full test sequence to align with prediction target
seq_start_idx = len(X_train)
true_velocity = df["velocity"].values[seq_start_idx + SEQ_LENGTH:seq_start_idx + SEQ_LENGTH + len(y_test)].astype(np.float32)

# Compute derivatives
true_acceleration = np.gradient(true_velocity, DT).astype(np.float32)
true_jerk = np.gradient(true_acceleration, DT).astype(np.float32)

pred_acceleration = np.gradient(y_pred, DT).astype(np.float32)
pred_jerk = np.gradient(pred_acceleration, DT).astype(np.float32)

time = df["time"].values[seq_start_idx + SEQ_LENGTH:seq_start_idx + SEQ_LENGTH + len(y_test)]

# Create subplots
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time, true_velocity, label="True Velocity")
plt.plot(time, y_pred, label="Predicted Velocity", linestyle="--")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, true_acceleration, label="True Acceleration")
plt.plot(time, pred_acceleration, label="Predicted Acceleration", linestyle="--")
plt.ylabel("Acceleration (m/s²)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, true_jerk, label="True Jerk")
plt.plot(time, pred_jerk, label="Predicted Jerk", linestyle="--")
plt.ylabel("Jerk (m/s³)")
plt.xlabel("Time (s)")
plt.legend()

plt.suptitle("Simple LSTM — Predicted vs True Profiles")
plt.tight_layout()
plot_path = os.path.join(plot_dir, "simple_LSTM_comparison_plot.png")
plt.savefig(plot_path)
plt.close()
logging.info(f"Saved full comparison plot to {plot_path}.")
