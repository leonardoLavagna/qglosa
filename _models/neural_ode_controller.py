import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

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

logging.basicConfig(filename=os.path.join(log_dir, "neural_ode_controller.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

######################################################
# Hyperparameters and constants
######################################################
SEQ_LENGTH = 50
DT = 0.1
a_tau_max = 2.0
a_nu_max = 2.5
u_max = 3.0
λ_vel = 10.0
λ_smooth = 1.0

FEATURE_COLUMNS = ['time', 'position', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]

######################################################
# Load data
######################################################
df = pd.read_csv("data/cleaned_BUS.csv")
scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATURE_COLUMNS + ["velocity"]])

def create_sequences(data, seq_len):
    X, v_target = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        v_target.append(data[i:i + seq_len, 2])  # column 2 = velocity
    return np.array(X, dtype=np.float32), np.array(v_target, dtype=np.float32)

X_all, v_all = create_sequences(scaled, SEQ_LENGTH)
X_train, X_test, v_train, v_test = train_test_split(X_all, v_all, test_size=0.2, random_state=42)

######################################################
# Curvature (mock)
######################################################
def curvature_tf(x):
    return tf.constant(0.01, dtype=tf.float32) * tf.sin(tf.constant(0.01, dtype=tf.float32) * x)

######################################################
# Neural controller
######################################################
class Controller(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        return tf.squeeze(self.out(x), axis=-1)  # shape (batch, seq)

model = Controller(input_dim=X_train.shape[2])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

######################################################
# Physics ODE Integrator (Euler)
######################################################
def integrate_physics(u_seq, x0, v0, a0, dt=DT):
    x, v, a = [x0], [v0], [a0]
    for t in range(u_seq.shape[1]):
        a_new = a[-1] + u_seq[:, t] * dt
        v_new = v[-1] + a_new * dt
        x_new = x[-1] + v_new * dt
        a.append(a_new)
        v.append(v_new)
        x.append(x_new)
    return tf.stack(x[1:], axis=1), tf.stack(v[1:], axis=1), tf.stack(a[1:], axis=1)

######################################################
# Training step
######################################################
@tf.function
def train_step(x_batch, v_target):
    with tf.GradientTape() as tape:
        u = model(x_batch)  # predicted jerk

        x0 = tf.zeros((x_batch.shape[0],), dtype=tf.float32)
        v0 = tf.random.uniform((x_batch.shape[0],), minval=1.0, maxval=3.0)
        a0 = tf.random.uniform((x_batch.shape[0],), minval=-0.5, maxval=0.5)

        x_sim, v_sim, a_sim = integrate_physics(u, x0, v0, a0)

        pos = x_batch[:, :, 1]
        kappa = curvature_tf(pos)
        a_nu = v_sim**2 * kappa

        v_loss = tf.reduce_mean(tf.square(v_sim - v_target))
        a_penalty = tf.reduce_mean(tf.nn.relu(tf.abs(a_sim) - a_tau_max))
        u_penalty = tf.reduce_mean(tf.nn.relu(tf.abs(u) - u_max))
        a_nu_penalty = tf.reduce_mean(tf.nn.relu(tf.abs(a_nu) - a_nu_max))
        u_smooth = tf.reduce_mean(tf.square(u[:, 1:] - u[:, :-1]))

        loss = λ_vel * v_loss + a_penalty + u_penalty + a_nu_penalty + λ_smooth * u_smooth

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, v_loss

######################################################
# Training loop
######################################################
EPOCHS = 30
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, v_train)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    epoch_loss, epoch_vloss = 0.0, 0.0
    for x_batch, v_target in train_dataset:
        loss, v_loss = train_step(x_batch, v_target)
        epoch_loss += loss.numpy()
        epoch_vloss += v_loss.numpy()
    logging.info(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, VelLoss = {epoch_vloss:.4f}")

model.save(os.path.join(model_dir, "neural_ode_controller.h5"))

######################################################
# Evaluation plot
######################################################
x_test_tensor = tf.convert_to_tensor(X_test)
u_pred = model(x_test_tensor)
x_sim, v_sim, a_sim = integrate_physics(u_pred, tf.zeros_like(u_pred[:, 0]), tf.zeros_like(u_pred[:, 0]), tf.zeros_like(u_pred[:, 0]))

v_pred = v_sim[0].numpy()
a_pred = a_sim[0].numpy()
jerk_pred = u_pred[0].numpy()

# True values from test batch
v_true = v_test[0]
a_true = np.gradient(v_true, DT).astype(np.float32)
jerk_true = np.gradient(a_true, DT).astype(np.float32)
time = df["time"].iloc[:SEQ_LENGTH].values

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

plt.suptitle("Neural ODE Controller — Predicted vs True Profiles")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "neural_ode_controller_comparison.png"))
plt.close()
