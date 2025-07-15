import pandas as pd
import os
import logging


######################################################
# Setup
######################################################
log_dir = "_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "preprocessing.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

column_names = ['time', 'position', 'velocity', 'acceleration'] + [f'light_{i}' for i in range(1, 12)]

def load_and_log(file_path, label):
    try:
        df = pd.read_csv(file_path, header=None, names=column_names)
        logging.info(f"{label} loaded successfully with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Failed to load {label}: {e}")
        raise


######################################################
# Data cleaning functions
######################################################
def check_and_clean(df, label):
    initial_shape = df.shape
    # Check missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logging.warning(f"{label}: Found {missing} missing values. Dropping rows with missing values.")
        df.dropna(inplace=True)
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logging.warning(f"{label}: Found {duplicate_count} duplicated rows. Dropping them.")
        df.drop_duplicates(inplace=True)
    # Check time monotonicity
    if not df['time'].is_monotonic_increasing:
        logging.warning(f"{label}: 'time' column not monotonic. Sorting.")
        df.sort_values('time', inplace=True)
    # Check stoplight columns are only 0 or 1
    for col in df.columns[4:]:
        invalid = ~df[col].isin([0, 1])
        if invalid.any():
            count = invalid.sum()
            logging.warning(f"{label}: Column '{col}' has {count} invalid stoplight values. Clipping to 0/1.")
            df[col] = df[col].clip(0, 1).astype(int)
    logging.info(f"{label}: Cleaned from shape {initial_shape} to {df.shape}.")
    return df

def save_clean_data(df, filename):
    output_path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned data to {output_path}")


######################################################
# Results
######################################################
def main():
    raw_dir = "_raw_data"
    df_bus = load_and_log(os.path.join(raw_dir, "data_training_BUS.csv"), "BUS")
    df_mpc = load_and_log(os.path.join(raw_dir, "data_training_MPC.csv"), "MPC")
    df_bus_clean = check_and_clean(df_bus, "BUS")
    df_mpc_clean = check_and_clean(df_mpc, "MPC")
    save_clean_data(df_bus_clean, "cleaned_BUS.csv")
    save_clean_data(df_mpc_clean, "cleaned_MPC.csv")

if __name__ == "__main__":
    main()
