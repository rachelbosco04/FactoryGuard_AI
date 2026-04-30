import pandas as pd
import os

# =========================
# CONFIG
# =========================
INPUT = "data/processed/synthetic_factory_data.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort globally (good practice)
df = df.sort_values(["machine_id", "timestamp"])

print("=" * 50)
print("Dataset Loaded")
print("Total rows:", len(df))
print("Total failures:", df["failure"].sum())
print("=" * 50)

# =========================
# SPLIT PER MACHINE
# =========================
train_list = []
val_list = []
test_list = []

for machine_id, group in df.groupby("machine_id"):
    group = group.sort_values("timestamp")

    n = len(group)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_list.append(group.iloc[:train_end])
    val_list.append(group.iloc[train_end:val_end])
    test_list.append(group.iloc[val_end:])

# Combine all machines
train = pd.concat(train_list).reset_index(drop=True)
val = pd.concat(val_list).reset_index(drop=True)
test = pd.concat(test_list).reset_index(drop=True)

# =========================
# SAVE SPLITS
# =========================
os.makedirs("data/processed", exist_ok=True)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

# =========================
# SUMMARY
# =========================
print("=" * 50)
print("Split Complete")
print("Train:", train.shape, "| Failures:", train["failure"].sum())
print("Val  :", val.shape, "| Failures:", val["failure"].sum())
print("Test :", test.shape, "| Failures:", test["failure"].sum())
print("=" * 50)