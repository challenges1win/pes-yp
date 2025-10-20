"""
collect_data.py
------------------------------------
Simulates or collects sensor data for:
 - Battery Banks
 - Solar Inverters
 - Transformers

Used in Phase 1 of the IEEE PES & YP Challenge.

In Phase 2, this script can be adapted to pull live readings
from IoT sensors, APIs, or SCADA systems instead of generating synthetic data.
------------------------------------
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------------------------------
# 🔧 CONFIGURATION
# -----------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

N_SAMPLES = 1000              # number of samples to generate initially
FREQ = "1H"                   # frequency: 1 hour
ANOMALY_RATIO = 0.025         # 2.5% anomalies injected


# -----------------------------------------------------------
# 🔋 BATTERY BANK
# -----------------------------------------------------------
def generate_battery_data(n_samples=N_SAMPLES, freq=FREQ):
    np.random.seed(42)
    time = pd.date_range(datetime(2025, 1, 1), periods=n_samples, freq=freq)
    df = pd.DataFrame({
        "time": time,
        "voltage": np.random.normal(48, 1, n_samples),
        "current": np.random.normal(10, 2, n_samples),
        "temperature": np.random.normal(30, 3, n_samples),
        "SoC": np.clip(np.random.normal(80, 10, n_samples), 0, 100),
        "cycle_count": np.random.randint(200, 1000, n_samples)
    })

    # Inject anomalies
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * ANOMALY_RATIO), replace=False)
    df.loc[anomaly_idx, "temperature"] += np.random.normal(15, 3, len(anomaly_idx))
    df.loc[anomaly_idx, "voltage"] -= np.random.normal(5, 1, len(anomaly_idx))
    df["anomaly"] = 0
    df.loc[anomaly_idx, "anomaly"] = 1

    # Random missing values
    for col in ["temperature", "voltage"]:
        df.loc[df.sample(frac=0.01).index, col] = np.nan

    path = os.path.join(DATA_DIR, "battery_bank.csv")
    df.to_csv(path, index=False)
    print(f"✅ Battery Bank data saved: {path}")
    return df


# -----------------------------------------------------------
# ☀️ SOLAR INVERTER
# -----------------------------------------------------------
def generate_solar_data(n_samples=N_SAMPLES, freq=FREQ):
    np.random.seed(43)
    time = pd.date_range(datetime(2025, 1, 1), periods=n_samples, freq=freq)
    df = pd.DataFrame({
        "time": time,
        "input_power": np.random.normal(5000, 500, n_samples),
        "output_power": np.random.normal(4800, 450, n_samples),
        "efficiency": np.clip(np.random.normal(0.95, 0.02, n_samples), 0.85, 1.0),
        "temperature": np.random.normal(40, 4, n_samples),
        "frequency": np.random.normal(50, 0.2, n_samples),
        "fault_code": np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    })

    # Inject anomalies
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * ANOMALY_RATIO), replace=False)
    df.loc[anomaly_idx, "efficiency"] -= np.random.normal(0.1, 0.03, len(anomaly_idx))
    df.loc[anomaly_idx, "temperature"] += np.random.normal(10, 2, len(anomaly_idx))
    df.loc[anomaly_idx, "fault_code"] = 1
    df["anomaly"] = 0
    df.loc[anomaly_idx, "anomaly"] = 1

    # Random missing values
    for col in ["input_power", "output_power"]:
        df.loc[df.sample(frac=0.01).index, col] = np.nan

    path = os.path.join(DATA_DIR, "solar_inverter.csv")
    df.to_csv(path, index=False)
    print(f"✅ Solar Inverter data saved: {path}")
    return df


# -----------------------------------------------------------
# ⚡ TRANSFORMER
# -----------------------------------------------------------
def generate_transformer_data(n_samples=N_SAMPLES, freq=FREQ):
    np.random.seed(44)
    time = pd.date_range(datetime(2025, 1, 1), periods=n_samples, freq=freq)
    df = pd.DataFrame({
        "time": time,
        "temperature": np.random.normal(60, 5, n_samples),
        "vibration": np.random.normal(0.5, 0.1, n_samples),
        "load_current": np.random.normal(100, 15, n_samples),
        "partial_discharge": np.random.normal(5, 2, n_samples)
    })

    # Inject anomalies
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * ANOMALY_RATIO), replace=False)
    df.loc[anomaly_idx, "temperature"] += np.random.normal(20, 5, len(anomaly_idx))
    df.loc[anomaly_idx, "vibration"] += np.random.normal(0.5, 0.2, len(anomaly_idx))
    df.loc[anomaly_idx, "partial_discharge"] += np.random.normal(5, 2, len(anomaly_idx))
    df["anomaly"] = 0
    df.loc[anomaly_idx, "anomaly"] = 1

    # Random missing values
    for col in ["temperature", "vibration"]:
        df.loc[df.sample(frac=0.01).index, col] = np.nan

    path = os.path.join(DATA_DIR, "transformer.csv")
    df.to_csv(path, index=False)
    print(f"✅ Transformer data saved: {path}")
    return df


# -----------------------------------------------------------
# 🧠 APPEND MODE FOR CONTINUOUS COLLECTION
# -----------------------------------------------------------
def append_new_data(sensor_name, df_new):
    """
    Appends new rows to existing sensor CSV file if it exists.
    """
    file_path = os.path.join(DATA_DIR, f"{sensor_name}.csv")
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path, parse_dates=["time"])
        combined = pd.concat([df_old, df_new], ignore_index=True)
        combined.drop_duplicates(subset=["time"], inplace=True)
        combined.to_csv(file_path, index=False)
        print(f"🔁 Appended new data to {file_path}")
    else:
        df_new.to_csv(file_path, index=False)
        print(f"🆕 Created new {file_path}")


# -----------------------------------------------------------
# 🚀 MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == "__main__":
    print("🔄 Generating sensor data...")
    battery_df = generate_battery_data()
    solar_df = generate_solar_data()
    transformer_df = generate_transformer_data()

    # Optionally append to existing files
    append_new_data("battery_bank", battery_df)
    append_new_data("solar_inverter", solar_df)
    append_new_data("transformer", transformer_df)

    print("\n✅ Data collection complete. Files saved in /data/")
