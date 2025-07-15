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

logging.basicConfig(filename=os.path.join(log_dir, "improved_data_driven_controller.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

######################################################
# Hyperparameters and constants
######################################################
SEQ_LENGTH = 50
DT = 0.1
λ_smooth = 1.0
λ_track = 10.0
λ_jerk = 1.0

FEATURE_COLUMNS = ['time', 'position', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]
TARGET_COLUMN = 'velocity'

######################################################
# Load and preprocess data
######################################################
df = pd.read_csv("data/cleaned_BUS.csv")
scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATURE_COLUMNS + [TARGET_COLUMN]])

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 2])  # index 2 = velocity
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = create_sequences(scaled, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

######################################################
# Physics simulator
######################################################
def physics_simulate(u, x0, v0, a0, dt=DT):
    a = tf.cumsum(u, axis=1) * dt + tf.expand_dims(a0, -1)
    v = tf.cumsum(a, axis=1) * dt + tf.expand_dims(v0, -1)
    x = tf.cumsum(v, axis=1) * dt + tf.expand_dims(x0, -1)
    return x, v, a

######################################################
# Neural controller model
######################################################
class Controller(tf.keras.Model):
    def __init__(self, seq_len, input_dim):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        return tf.squeeze(self.out(x), axis=-1)

model = Controller(seq_len=SEQ_LENGTH, input_dim=X_train.shape[2])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

######################################################
# Prepare jerk supervision target
######################################################
acc_true = df["acceleration"].values.astype(np.float32)
jerk_all = np.gradient(acc_true, DT).astype(np.float32)
jerk_seq = jerk_all[:len(X_all)].reshape(-1, 1)
jerk_seq = np.tile(jerk_seq, (1, SEQ_LENGTH))[:len(X_all)]

######################################################
# Training step
######################################################
@tf.function
def train_step(x_batch, jerk_true_batch, v_target_batch):
    with tf.GradientTape() as tape:
        u_pred = model(x_batch, training=True)

        # Random initial conditions
        x0 = tf.zeros((x_batch.shape[0],), dtype=tf.float32)
        v0 = tf.random.uniform((x_batch.shape[0],), minval=1.0, maxval=3.0, dtype=tf.float32)
        a0 = tf.random.uniform((x_batch.shape[0],), minval=-0.5, maxval=0.5, dtype=tf.float32)

        x_sim, v_sim, a_sim = physics_simulate(u_pred, x0, v0, a0)

        # Stoplight penalty
        light_states = x_batch[:, :, 3:]
        penalty = 0.0
        for i in range(11):
            red_mask = 1.0 - light_states[:, :, i]
            penalty += tf.reduce_mean(tf.square(v_sim) * red_mask)

        velocity_tracking_loss = tf.reduce_mean(tf.square(v_sim[:, -1] - v_target_batch))
        jerk_loss = tf.reduce_mean(tf.square(u_pred - jerk_true_batch))
        jerk_smooth = tf.reduce_mean(tf.square(u_pred[:, 1:] - u_pred[:, :-1]))

        loss = penalty + λ_track * velocity_tracking_loss + λ_jerk * jerk_loss + λ_smooth * jerk_smooth

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

######################################################
# Training loop
######################################################
EPOCHS = 30
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, jerk_seq[:len(X_train)].astype(np.float32), y_train)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    epoch_loss = 0
    for step, (x_batch, jerk_true_batch, v_target_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, jerk_true_batch, v_target_batch)
        epoch_loss += loss.numpy()
    logging.info(f"Epoch {epoch+1} — Loss: {epoch_loss:.4f}")

model.save(os.path.join(model_dir, "improved_data_driven_controller.h5"))

######################################################
# Evaluation plot
######################################################
X_test_tensor = tf.convert_to_tensor(X_test)
u_pred = model(X_test_tensor, training=False)
x_sim, v_sim, a_sim = physics_simulate(u_pred, tf.zeros_like(u_pred[:, 0]), tf.zeros_like(u_pred[:, 0]), tf.zeros_like(u_pred[:, 0]))

v_pred = v_sim[0].numpy()
a_pred = a_sim[0].numpy()
jerk_pred = u_pred[0].numpy()

start_idx = 0
end_idx = start_idx + SEQ_LENGTH
v_true = df["velocity"].iloc[start_idx:end_idx].values.astype(np.float32)
a_true = df["acceleration"].iloc[start_idx:end_idx].values.astype(np.float32)
jerk_true = np.gradient(a_true, DT).astype(np.float32)
time = df["time"].iloc[start_idx:end_idx].values

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time, v_true, label="True Velocity")
plt.plot(time, v_pred, label="Predicted Velocity", linestyle="--")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, a_true, label="True Acceleration")
plt.plot(time, a_pred, label="Predicted Acceleration", linestyle="--")
plt.ylabel("Acceleration (m/s²)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, jerk_true, label="True Jerk")
plt.plot(time, jerk_pred, label="Predicted Jerk", linestyle="--")
plt.ylabel("Jerk (m/s³)")
plt.xlabel("Time (s)")
plt.legend()

plt.suptitle("Improved Controller — Predicted vs True Profiles")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "improved_controller_comparison.png"))
plt.close()
