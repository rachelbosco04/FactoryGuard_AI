import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import lightgbm as lgb

# ============================================================
# PATHS
# ============================================================
FEATURES_DIR = os.path.join("outputs", "week1_features")
MODELS_DIR   = "models"
REPORTS_DIR  = os.path.join("reports", "week3_shap")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "failure"
DROP_COLS = ["timestamp", "machine_id", TARGET]


# ============================================================
# CLEAN FEATURE NAMES (MANDATORY FOR LIGHTGBM)
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
    print("Loading engineered datasets ...")

    train = joblib.load(os.path.join(FEATURES_DIR, "train_features.joblib"))
    val   = joblib.load(os.path.join(FEATURES_DIR, "val_features.joblib"))
    test  = joblib.load(os.path.join(FEATURES_DIR, "test_features.joblib"))

    feat_cols = [c for c in train.columns if c not in DROP_COLS]

    X_train = train[feat_cols].copy()
    y_train = train[TARGET].astype(int)

    X_val = val[feat_cols].copy()
    y_val = val[TARGET].astype(int)

    X_test = test[feat_cols].copy()
    y_test = test[TARGET].astype(int)

    # Fix names
    X_train = clean_feature_names(X_train)
    X_val   = clean_feature_names(X_val)
    X_test  = clean_feature_names(X_test)

    print(f"Train : {X_train.shape} | Failures: {y_train.sum():,}")
    print(f"Val   : {X_val.shape}   | Failures: {y_val.sum():,}")
    print(f"Test  : {X_test.shape}  | Failures: {y_test.sum():,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# SMOTE
# ============================================================
def apply_smote(X_train, y_train, sampling_strategy=0.10):
    print("\nApplying SMOTE ...")
    print("Before:", pd.Series(y_train).value_counts().to_dict())

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=5
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("After :", pd.Series(y_res).value_counts().to_dict())
    print("Shape :", X_res.shape)

    return X_res, y_res


# ============================================================
# TRAIN LIGHTGBM
# ============================================================
def train_lgbm(X_train, y_train, X_val, y_val, title="Model"):
    print(f"\nTraining {title} ...")

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    return model


# ============================================================
# EVALUATE
# ============================================================
def evaluate(name, model, X, y):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y, proba)
    roc = roc_auc_score(y, proba)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print("\n" + "="*60)
    print(name)
    print("="*60)
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"ROC-AUC   : {roc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, zero_division=0))

    save_pr_curve(name, y, proba, pr_auc)

    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# SAVE PR CURVE
# ============================================================
def save_pr_curve(name, y_true, y_prob, ap):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fname = name.lower().replace(" ", "_")
    path = os.path.join(REPORTS_DIR, f"pr_curve_{fname}.png")

    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(name)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ============================================================
# SHAP
# ============================================================
def run_shap(model, X_test):
    print("\nRunning SHAP Explainability ...")

    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary beeswarm
    path1 = os.path.join(REPORTS_DIR, "shap_summary.png")
    shap.summary_plot(shap_vals, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path1, dpi=140, bbox_inches="tight")
    plt.close()

    # Bar plot
    path2 = os.path.join(REPORTS_DIR, "shap_bar.png")
    shap.summary_plot(
        shap_vals,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(path2, dpi=140, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(path1)
    print(path2)


# ============================================================
# TABLE
# ============================================================
def print_table(results):
    print("\n" + "="*72)
    print("WEEK 3 COMPARISON")
    print("="*72)
    print(f"{'Model':30} {'PR-AUC':>8} {'ROC':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")

    for r in results:
        print(
            f"{r['model']:30} "
            f"{r['pr_auc']:>8.4f} "
            f"{r['roc_auc']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f}"
        )


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*65)
    print("FactoryGuard AI — Week 3 Imbalance + SHAP")
    print("="*65)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = []

    # --------------------------------------------------------
    # Model 1: Weighted
    # --------------------------------------------------------
    weighted = train_lgbm(
        X_train, y_train, X_val, y_val,
        title="LightGBM Weighted"
    )

    res = evaluate("LightGBM Weighted", weighted, X_test, y_test)
    results.append(res)

    joblib.dump(
        weighted,
        os.path.join(MODELS_DIR, "week3_lgbm_weighted.joblib"),
        compress=3
    )

    # --------------------------------------------------------
    # Model 2: SMOTE
    # --------------------------------------------------------
    X_sm, y_sm = apply_smote(X_train, y_train, 0.10)

    smote_model = train_lgbm(
        X_sm, y_sm, X_val, y_val,
        title="LightGBM SMOTE"
    )

    res = evaluate("LightGBM SMOTE", smote_model, X_test, y_test)
    results.append(res)

    joblib.dump(
        smote_model,
        os.path.join(MODELS_DIR, "week3_lgbm_smote.joblib"),
        compress=3
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print_table(results)

    # --------------------------------------------------------
    # SHAP on weighted model
    # --------------------------------------------------------
    run_shap(weighted, X_test)

    print("\nWeek 3 Complete.")
    print(f"Outputs saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()