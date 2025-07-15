import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

logging.basicConfig(filename=os.path.join(log_dir, "physics_informed_LSTM.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


######################################################
# Hyperparameters and constants
######################################################
SEQ_LENGTH = 20
DT = 0.1  # time step (s)

FEATURE_COLUMNS = ['time', 'position', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]
TARGET_COLUMN = 'velocity'

# Physical limits for constraints
a_tau_max = 2.0      # max longitudinal acceleration (m/s²)
u_max = 3.0          # max jerk (m/s³)
a_nu_max = 2.5       # max lateral acceleration (m/s²)

# Penalty weights
λ1 = 10.0  # acceleration
λ2 = 5.0   # jerk
λ3 = 5.0   # lateral acceleration


######################################################
# Utility functions
######################################################
def curvature_tf(x):
    # TensorFlow-compatible mock curvature profile
    return tf.constant(0.01, dtype=tf.float32) * tf.sin(tf.constant(0.01, dtype=tf.float32) * x)

def create_sequences(data, seq_length):
    # Converts a flat time-series to overlapping sequences of length `seq_length`
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 2])  # velocity at t+1
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


######################################################
# Load and prepare data
######################################################
df = pd.read_csv("data/cleaned_BUS.csv")
logging.info(f"Loaded data with shape: {df.shape}")

scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATURE_COLUMNS + [TARGET_COLUMN]])
X_all, y_all = create_sequences(scaled, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
logging.info(f"Prepared sequences: X_train={X_train.shape}, y_train={y_train.shape}")


######################################################
# Model definition
######################################################
class PhysicsLSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        return tf.squeeze(self.dense(x), axis=-1)  # shape: (batch, seq_length)

model = PhysicsLSTM()
optimizer = tf.keras.optimizers.Adam()


######################################################
# Custom Physics-Informed Training Step
######################################################
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        v = model(x_batch, training=True)  # predicted velocity sequence
        # Compute physics-based terms
        a = (v[:, 1:] - v[:, :-1]) / DT
        u = (a[:, 1:] - a[:, :-1]) / DT
        a = tf.concat([a, tf.zeros_like(a[:, :1])], axis=1)
        u = tf.concat([u, tf.zeros_like(u[:, :2])], axis=1)
        pos = x_batch[:, :, 1]  
        kappa = curvature_tf(pos)  
        a_nu = v**2 * kappa  
        # Compute loss terms
        acc_penalty = tf.reduce_mean(tf.nn.relu(tf.abs(a) - a_tau_max))
        jerk_penalty = tf.reduce_mean(tf.nn.relu(tf.abs(u) - u_max))
        lat_penalty = tf.reduce_mean(tf.nn.relu(tf.abs(a_nu) - a_nu_max))
        mse = tf.reduce_mean(tf.square(y_batch - v[:, -1]))  # predict final velocity
        # Total loss
        loss = mse + λ1 * acc_penalty + λ2 * jerk_penalty + λ3 * lat_penalty

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, mse


######################################################
# Training loop
######################################################
EPOCHS = 20
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    epoch_loss, epoch_mse = 0.0, 0.0
    for x_batch, y_batch in train_dataset:
        loss, mse = train_step(x_batch, y_batch)
        epoch_loss += loss.numpy()
        epoch_mse += mse.numpy()
    logging.info(f"Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss:.4f}, MSE: {epoch_mse:.4f}")


######################################################
# Evaluation
######################################################
X_test_tensor = tf.convert_to_tensor(X_test)
y_pred_seq = model(X_test_tensor, training=False).numpy()
y_pred_last = y_pred_seq[:, -1]
mse = np.mean((y_test - y_pred_last) ** 2)
# Save evaluation result
results_path = os.path.join(results_dir, "physics_informed_LSTM.txt")
with open(results_path, "w") as f:
    f.write(f"Test MSE: {mse:.4f}\n")
logging.info(f"Saved test MSE to {results_path}")


######################################################
# Save model
######################################################
model_path = os.path.join(model_dir, "physics_informed_LSTM.h5")
model.save(model_path)
logging.info(f"Saved trained model to {model_path}")


######################################################
# Plotting
######################################################
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="True Velocity", alpha=0.7)
plt.plot(y_pred_last, label="Predicted Velocity", alpha=0.7)
plt.xlabel("Sample")
plt.ylabel("Velocity (m/s)")
plt.title("Physics-Informed LSTM Velocity Prediction")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(plot_dir, "physics_informed_LSTM_velocity_prediction_plot.png")
plt.savefig(plot_path)
plt.close()
logging.info(f"Saved plot to {plot_path}")
