import pandas as pd
import numpy as np
import joblib
import os
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data"
PROCESSED   = os.path.join(DATA_DIR, "processed")
FEATURES = os.path.join("outputs", "week1_features")
os.makedirs(FEATURES, exist_ok=True)

TRAIN_PATH  = os.path.join(PROCESSED, "train.csv")
VAL_PATH    = os.path.join(PROCESSED, "val.csv")
TEST_PATH   = os.path.join(PROCESSED, "test.csv")

os.makedirs(FEATURES, exist_ok=True)

# Sensor columns to engineer features on
SENSOR_COLS = [
    "air_temperature_[k]",
    "process_temperature_[k]",
    "rotational_speed_[rpm]",
    "torque_[nm]",
    "tool_wear_[min]",
]

# Rolling window sizes (hourly data → 1h=1, 6h=6, 12h=12)
WINDOWS = [1, 6, 12]

# Lag steps
LAG_STEPS = [1, 2]


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_splits():
    print("Loading train / val / test splits …")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["timestamp"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["timestamp"])

    print(f"  Train : {train.shape}")
    print(f"  Val   : {val.shape}")
    print(f"  Test  : {test.shape}")
    return train, val, test


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
    """
    Applies per-machine, time-ordered feature engineering:
      1. Rolling Mean, EMA (Exponential Moving Average), Std Dev — windows: 1h, 6h, 12h
      2. Lag features — t-1, t-2
    """
    print(f"\n[{split_name}] Engineering features on {len(df):,} rows …")
    t0 = time.time()

    # Sort by machine then time so rolling/lag ops are chronologically correct
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    engineered_chunks = []

    for machine_id, grp in df.groupby("machine_id", sort=False):
        grp = grp.copy()

        for col in SENSOR_COLS:
            safe = col.replace("[", "").replace("]", "").replace(" ", "_")

            # ── Rolling Statistics ────────────────────────────────────────
            for w in WINDOWS:
                # Rolling Mean
                grp[f"{safe}_roll_mean_{w}h"] = (
                    grp[col].rolling(window=w, min_periods=1).mean()
                )

                # Exponential Moving Average (span = window)
                grp[f"{safe}_ema_{w}h"] = (
                    grp[col].ewm(span=w, adjust=False).mean()
                )

                # Rolling Standard Deviation
                grp[f"{safe}_roll_std_{w}h"] = (
                    grp[col].rolling(window=w, min_periods=1).std().fillna(0)
                )

            # ── Lag Features ──────────────────────────────────────────────
            for lag in LAG_STEPS:
                grp[f"{safe}_lag_{lag}"] = grp[col].shift(lag)

        engineered_chunks.append(grp)

    result = pd.concat(engineered_chunks, ignore_index=True)

    # Fill any NaNs introduced by lagging the very first rows per machine
    lag_cols = [c for c in result.columns if "_lag_" in c]
    result[lag_cols] = result[lag_cols].bfill().fillna(0)

    elapsed = time.time() - t0
    new_features = result.shape[1] - df.shape[1]
    print(f"  ✓ Done in {elapsed:.1f}s | Shape: {result.shape} | +{new_features} new features")
    return result


# ─────────────────────────────────────────────
# SERIALIZE WITH JOBLIB
# ─────────────────────────────────────────────
def save_features(df: pd.DataFrame, name: str):
    """Serialize engineered DataFrame to joblib for fast I/O."""
    path = os.path.join(FEATURES, f"{name}_features.joblib")
    joblib.dump(df, path, compress=3)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  💾 Saved → {path}  ({size_mb:.1f} MB)")


def load_features(name: str) -> pd.DataFrame:
    """Load serialized feature DataFrame."""
    path = os.path.join(FEATURES, f"{name}_features.joblib")
    df = joblib.load(path)
    print(f"  📂 Loaded {name}: {df.shape}")
    return df


# ─────────────────────────────────────────────
# FEATURE SUMMARY REPORT
# ─────────────────────────────────────────────
def print_feature_summary(df: pd.DataFrame):
    roll_cols = [c for c in df.columns if "_roll_" in c]
    ema_cols  = [c for c in df.columns if "_ema_"  in c]
    lag_cols  = [c for c in df.columns if "_lag_"  in c]

    print("\n" + "="*60)
    print("  FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"  Total columns          : {df.shape[1]}")
    print(f"  Rolling Mean features  : {len([c for c in roll_cols if 'mean' in c])}")
    print(f"  Rolling Std features   : {len([c for c in roll_cols if 'std'  in c])}")
    print(f"  EMA features           : {len(ema_cols)}")
    print(f"  Lag features           : {len(lag_cols)}")
    print(f"  Null values            : {df.isnull().sum().sum()}")
    print("="*60)

    print("\n  Sample engineered columns:")
    engineered = [c for c in df.columns if any(
        x in c for x in ["_roll_", "_ema_", "_lag_"]
    )]
    for c in engineered[:12]:
        print(f"    • {c}")
    if len(engineered) > 12:
        print(f"    … and {len(engineered) - 12} more")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FactoryGuard AI — Week 1: Feature Engineering")
    print("=" * 60)

    train, val, test = load_splits()

    # Engineer features for each split
    train_fe = engineer_features(train, split_name="TRAIN")
    val_fe   = engineer_features(val,   split_name="VAL")
    test_fe  = engineer_features(test,  split_name="TEST")

    # Serialize with joblib
    print("\nSerializing engineered datasets …")
    save_features(train_fe, "train")
    save_features(val_fe,   "val")
    save_features(test_fe,  "test")

    # Summary
    print_feature_summary(train_fe)

    print("\n✅ Feature engineering complete. Ready for LightGBM training.")


if __name__ == "__main__":
    main()