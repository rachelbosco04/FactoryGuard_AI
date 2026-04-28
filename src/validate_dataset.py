import pandas as pd

df = pd.read_csv("data/processed/synthetic_factory_data.csv")

print("=" * 50)
print("Dataset Shape:", df.shape)
print("Machines:", df["machine_id"].nunique())
print("Failures:", df["failure"].sum())
print("Null Values:")
print(df.isnull().sum())
print("=" * 50)