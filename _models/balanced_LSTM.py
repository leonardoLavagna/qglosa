import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model


######################################################
# Setup
######################################################
log_dir = "_logs"
model_dir = "_models"
plot_dir = "plots"
results_dir = "results"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, "balanced_LSTM.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


######################################################
# Parameters
######################################################
SEQ_LENGTH = 20
DT = 0.1
FEATURE_COLUMNS = ['time', 'position', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]
TARGET_COLUMN = 'velocity'


######################################################
# Load both datasets
######################################################
df_mpc = pd.read_csv("data/cleaned_MPC.csv")
df_bus = pd.read_csv("data/cleaned_BUS.csv")

# Add source label for tracking (not used in model input directly here)
df_mpc["source"] = "mpc"
df_bus["source"] = "bus"

# Combine and shuffle
df_combined = pd.concat([df_mpc, df_bus], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


######################################################
# Preprocessing
######################################################
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_combined[FEATURE_COLUMNS + [TARGET_COLUMN]])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 2]) 
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = create_sequences(scaled_data, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


######################################################
# Model
######################################################
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
logging.info("Compiled hybrid LSTM model.")

history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=0)
logging.info("Trained hybrid LSTM model.")

# Predict
y_pred = model.predict(X_test).flatten()
mse = np.mean((y_test - y_pred) ** 2)
logging.info(f"Hybrid Model — MSE: {mse:.4f}")


######################################################
# Save results
######################################################
results_path = os.path.join(results_dir, "balanced_LSTM.txt")
with open(results_path, "w") as f:
    f.write(f"Mean Squared Error: {mse:.4f}\n")
logging.info(f"Saved evaluation to {results_path}.")

model_path = os.path.join(model_dir, "balanced_LSTM.h5")
model.save(model_path)
logging.info(f"Saved trained model to {model_path}.")

######################################################
# Plot
######################################################
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="True Velocity", alpha=0.7)
plt.plot(y_pred, label="Predicted Velocity", linestyle="--", alpha=0.7)
plt.xlabel("Sample")
plt.ylabel("Velocity (m/s)")
plt.title("Hybrid LSTM Velocity Prediction — MPC + BUS")
plt.legend()
plot_path = os.path.join(plot_dir, "balanced_LSTM_velocity_prediction_plot.png")
plt.savefig(plot_path)
plt.close()
logging.info(f"Saved hybrid velocity plot to {plot_path}.")
