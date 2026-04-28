import os
import re
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
FEATURES_DIR = os.path.join("outputs", "week1_features")
MODELS_DIR = "models"
REPORTS_DIR = "reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "failure"
DROP_COLS = ["timestamp", "machine_id", TARGET]


# ============================================================
# CLEAN FEATURE NAMES (Fixes LightGBM JSON char issue)
# ============================================================
def clean_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        for col in df.columns
    ]
    return df


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    print("Loading engineered feature sets ...")

    train = joblib.load(os.path.join(FEATURES_DIR, "train_features.joblib"))
    val = joblib.load(os.path.join(FEATURES_DIR, "val_features.joblib"))
    test = joblib.load(os.path.join(FEATURES_DIR, "test_features.joblib"))

    def split_xy(df):
        feature_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feature_cols].copy()
        y = df[TARGET].astype(int)
        return X, y

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    # IMPORTANT FIX
    X_train = clean_feature_names(X_train)
    X_val = clean_feature_names(X_val)
    X_test = clean_feature_names(X_test)

    print(f"Train : {X_train.shape} | Failures: {y_train.sum():,}")
    print(f"Val   : {X_val.shape}   | Failures: {y_val.sum():,}")
    print(f"Test  : {X_test.shape}  | Failures: {y_test.sum():,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# METRICS
# ============================================================
def evaluate(name, model, X, y, threshold=0.5):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        roc = roc_auc_score(y, proba)
    else:
        preds = model.predict(X)
        roc = None

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    if roc is not None:
        print(f"ROC-AUC   : {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y, preds, zero_division=0))

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
    }


# ============================================================
# CLASS WEIGHT
# ============================================================
def compute_scale_pos_weight(y):
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos


# ============================================================
# BASELINE 1 — LOGISTIC REGRESSION
# ============================================================
def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression ...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=42
        ))
    ])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"Done in {time.time()-t0:.1f}s")
    return pipe


# ============================================================
# BASELINE 2 — RANDOM FOREST
# ============================================================
def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest ...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Done in {time.time()-t0:.1f}s")
    return model


# ============================================================
# PRODUCTION MODEL — LIGHTGBM
# ============================================================
def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\nTraining LightGBM ...")

    scale_pos = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight = {scale_pos:.1f}")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    t0 = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    print(f"Done in {time.time()-t0:.1f}s")
    return model


# ============================================================
# SAVE MODEL
# ============================================================
def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path, compress=3)
    print(f"Saved -> {path}")


# ============================================================
# RESULTS TABLE
# ============================================================
def print_results(results):
    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(f"{'Model':30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'ROC':>8}")

    for r in results:
        roc = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        print(
            f"{r['model']:30} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{roc:>8}"
        )


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("FactoryGuard AI — Week 2 Modeling Pipeline")
    print("=" * 65)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = []

    # Logistic Regression
    lr = train_logistic_regression(X_train, y_train)
    results.append(evaluate("Logistic Regression (Val)", lr, X_val, y_val))
    save_model(lr, "baseline_logistic_regression")

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    results.append(evaluate("Random Forest (Val)", rf, X_val, y_val))
    save_model(rf, "baseline_random_forest")

    # LightGBM
    lgbm = train_lightgbm(X_train, y_train, X_val, y_val)
    results.append(evaluate("LightGBM (Val)", lgbm, X_val, y_val))
    save_model(lgbm, "lightgbm_production")

    # Final Test
    print("\nFinal Test Evaluation:")
    evaluate("LightGBM (Test)", lgbm, X_test, y_test)

    print_results(results)

    print("\nWeek 2 Complete.")


if __name__ == "__main__":
    main()