import pandas as pd
import os

INPUT = "data/processed/synthetic_factory_data.csv"

df = pd.read_csv(INPUT)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

os.makedirs("data/processed", exist_ok=True)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("=" * 50)
print("Split Complete")
print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)
print("=" * 50)