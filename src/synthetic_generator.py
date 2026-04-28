import pandas as pd
import numpy as np
import os

np.random.seed(42)

INPUT_PATH = "data/raw/ai4i2020.csv"
OUTPUT_PATH = "data/processed/synthetic_factory_data.csv"

df = pd.read_csv(INPUT_PATH)

# Clean columns
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

sensor_cols = [
    "air_temperature_[k]",
    "process_temperature_[k]",
    "rotational_speed_[rpm]",
    "torque_[nm]",
    "tool_wear_[min]"
]

base = df[sensor_cols].copy()

NUM_MACHINES = 500
HOURS = 720   # 30 days hourly

all_data = []

for i in range(1, NUM_MACHINES + 1):

    machine_id = f"M{i:03d}"

    sample = base.sample(HOURS, replace=True).reset_index(drop=True)

    sample["timestamp"] = pd.date_range(
        start="2025-01-01",
        periods=HOURS,
        freq="h"
    )

    sample["machine_id"] = machine_id

    # gradual wear drift
    sample["tool_wear_[min]"] += np.arange(HOURS) * np.random.uniform(0.01, 0.05)

    # noise
    sample["air_temperature_[k]"] += np.random.normal(0, 0.5, HOURS)
    sample["process_temperature_[k]"] += np.random.normal(0, 0.6, HOURS)
    sample["rotational_speed_[rpm]"] += np.random.normal(0, 12, HOURS)
    sample["torque_[nm]"] += np.random.normal(0, 1.2, HOURS)

    # failures
    fail = (
        (sample["tool_wear_[min]"] > 220) &
        (sample["torque_[nm]"] > 50)
    )

    sample["failure"] = np.where(fail, 1, 0)

    random_fail = np.random.choice(HOURS, size=3, replace=False)
    sample.loc[random_fail, "failure"] = 1

    all_data.append(sample)

final_df = pd.concat(all_data, ignore_index=True)

os.makedirs("data/processed", exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 50)
print("Synthetic Dataset Created")
print("Rows:", len(final_df))
print("Machines:", final_df["machine_id"].nunique())
print("Failures:", final_df["failure"].sum())
print("=" * 50)