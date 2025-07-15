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

logging.basicConfig(filename=os.path.join(log_dir, "data_driven_controller.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


######################################################
# Hyperparameters and constants
######################################################
SEQ_LENGTH = 50
DT = 0.1
λ_smooth = 1.0

FEATURE_COLUMNS = ['time', 'position', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]


######################################################
# Load data
######################################################
df = pd.read_csv("data/cleaned_BUS.csv")
scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATURE_COLUMNS])

def create_sequences(data, seq_len):
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
    return np.array(X, dtype=np.float32)

X_all = create_sequences(scaled, SEQ_LENGTH)
X_train, X_test = train_test_split(X_all, test_size=0.2, random_state=42)


######################################################
# Physics integrator (jerk → acceleration → velocity → position)
######################################################
def physics_simulate(u, x0, v0, a0, dt=DT):
    a = tf.cumsum(u, axis=1) * dt + a0[:, None]
    v = tf.cumsum(a, axis=1) * dt + v0[:, None]
    x = tf.cumsum(v, axis=1) * dt + x0[:, None]
    return x, v, a


######################################################
# Neural controller model (predicts jerk profile)
######################################################
class Controller(tf.keras.Model):
    def __init__(self, seq_len, input_dim):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        return tf.squeeze(self.out(x), axis=-1)  # output jerk

model = Controller(seq_len=SEQ_LENGTH, input_dim=X_train.shape[2])
optimizer = tf.keras.optimizers.Adam()


######################################################
# Loss and training loop
######################################################
@tf.function
def train_step(x_batch):
    with tf.GradientTape() as tape:
        u_pred = model(x_batch, training=True)
        x0 = tf.zeros((x_batch.shape[0],), dtype=tf.float32)
        v0 = tf.zeros_like(x0)
        a0 = tf.zeros_like(x0)
        x_sim, v_sim, a_sim = physics_simulate(u_pred, x0, v0, a0)
        light_states = x_batch[:, :, 3:]
        penalty = 0.0
        for i in range(11):
            red_mask = 1.0 - light_states[:, :, i]
            penalty += tf.reduce_mean(tf.square(v_sim) * red_mask)
        jerk_smooth = tf.reduce_mean(tf.square(u_pred[:, 1:] - u_pred[:, :-1]))
        loss = penalty + λ_smooth * jerk_smooth
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


######################################################
# Training
######################################################
EPOCHS = 20
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    epoch_loss = 0
    for step, x_batch in enumerate(train_dataset):
        loss = train_step(x_batch)
        epoch_loss += loss.numpy()
    logging.info(f"Epoch {epoch+1} — Loss: {epoch_loss:.4f}")


######################################################
# Evaluation: Simulate one test sample
######################################################
X_test_tensor = tf.convert_to_tensor(X_test)
u_test = model(X_test_tensor, training=False)
x_sim, v_sim, a_sim = physics_simulate(u_test, tf.zeros_like(u_test[:, 0]), tf.zeros_like(u_test[:, 0]), tf.zeros_like(u_test[:, 0]))

# Choose first sample for plotting
v_pred = v_sim[0].numpy()
a_pred = a_sim[0].numpy()
jerk_pred = u_test[0].numpy()

# Ground truth from dataset (same index range)
start_idx = 0
end_idx = start_idx + SEQ_LENGTH
v_true = df["velocity"].iloc[start_idx:end_idx].values
a_true = df["acceleration"].iloc[start_idx:end_idx].values
jerk_true = np.gradient(a_true, edge_order=2)
time = df["time"].iloc[start_idx:end_idx].values


######################################################
# Plot: Comparison
######################################################
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(time, v_true, label="True Velocity")
axs[0].plot(time, v_pred, label="Predicted Velocity", linestyle="--")
axs[0].set_ylabel("Velocity (m/s)")
axs[0].legend()

axs[1].plot(time, a_true, label="True Acceleration")
axs[1].plot(time, a_pred, label="Predicted Acceleration", linestyle="--")
axs[1].set_ylabel("Acceleration (m/s²)")
axs[1].legend()

axs[2].plot(time, jerk_true, label="True Jerk")
axs[2].plot(time, jerk_pred, label="Predicted Jerk", linestyle="--")
axs[2].set_ylabel("Jerk (m/s³)")
axs[2].set_xlabel("Time (s)")
axs[2].legend()

plt.suptitle("Predicted vs True Profiles (Velocity, Acceleration, Jerk)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "data_driven_controller_comparison.png"))
plt.close()

model.save(os.path.join(model_dir, "data_driven_controller.h5"))
logging.info("Saved model and comparison plot.")
