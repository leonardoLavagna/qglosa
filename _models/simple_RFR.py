import pandas as pd
import os
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


######################################################
# Setup
######################################################
log_dir = "_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "simple_RFR.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

model_dir = "_models"
os.makedirs(model_dir, exist_ok=True)

plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


######################################################
# Data
######################################################
data_path = os.path.join("data", "cleaned_MPC.csv")
df = pd.read_csv(data_path)
logging.info(f"Loaded training data from {data_path} with shape {df.shape}.")
# Features and target
X = df.drop(columns=["velocity"])
y = df["velocity"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Split data into training and testing sets.")


######################################################
# Model
######################################################
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
logging.info("Trained RandomForestRegressor model.")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logging.info(f"Model evaluation — MSE: {mse:.4f}, R2: {r2:.4f}")


######################################################
# Results
######################################################
results_path = os.path.join(results_dir, "simple_RFR.txt")
with open(results_path, "w") as f:
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
logging.info(f"Saved model evaluation to {results_path}.")

model_path = os.path.join(model_dir, "simple_RFR.pkl")
joblib.dump(model, model_path)
logging.info(f"Saved trained model to {model_path}.")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="True Velocity", alpha=0.7)
plt.plot(y_pred, label="Predicted Velocity", alpha=0.7)
plt.xlabel("Sample")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Prediction vs True")
plt.legend()
plot_path = os.path.join(plot_dir, "simple_RFR_velocity_prediction_plot.png")
plt.savefig(plot_path)
plt.close()
logging.info(f"Saved velocity prediction plot to {plot_path}.")