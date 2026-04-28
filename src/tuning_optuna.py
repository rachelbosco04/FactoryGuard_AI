import os
import re
import joblib
import optuna
import lightgbm as lgb
import pandas as pd

from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# PATHS
# ============================================================
FEATURES_DIR = os.path.join("outputs", "week1_features")
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "failure"
DROP_COLS = ["timestamp", "machine_id", TARGET]

N_TRIALS = 50   # Increase to 100+ for deeper tuning


# ============================================================
# CLEAN FEATURE NAMES
# Fixes LightGBM special character issue
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
    val = joblib.load(os.path.join(FEATURES_DIR, "val_features.joblib"))

    feat_cols = [c for c in train.columns if c not in DROP_COLS]

    X_train = train[feat_cols].copy()
    y_train = train[TARGET].astype(int)

    X_val = val[feat_cols].copy()
    y_val = val[TARGET].astype(int)

    # IMPORTANT FIX
    X_train = clean_feature_names(X_train)
    X_val = clean_feature_names(X_val)

    print(f"Train Shape : {X_train.shape}")
    print(f"Val Shape   : {X_val.shape}")
    print(f"Failures    : {y_train.sum():,}")

    return X_train, y_train, X_val, y_val


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================
def make_objective(X_train, y_train, X_val, y_val, scale_pos_weight):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

            "scale_pos_weight": scale_pos_weight,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(30, verbose=False)
            ],
        )

        preds = model.predict(X_val)

        score = f1_score(y_val, preds, zero_division=0)

        return score

    return objective


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("FactoryGuard AI — Optuna Hyperparameter Tuning")
    print("=" * 65)

    X_train, y_train, X_val, y_val = load_data()

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nscale_pos_weight = {scale_pos:.1f}")

    study = optuna.create_study(
        direction="maximize",
        study_name="factoryguard_lgbm",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    print(f"\nRunning {N_TRIALS} trials ...")

    study.optimize(
        make_objective(X_train, y_train, X_val, y_val, scale_pos),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial

    print("\n" + "=" * 65)
    print("BEST RESULT")
    print("=" * 65)
    print(f"Best Trial : #{best.number}")
    print(f"Best F1    : {best.value:.4f}")
    print("Best Params:")

    for k, v in best.params.items():
        print(f"{k:<22}: {v}")

    # ========================================================
    # RETRAIN FINAL MODEL
    # ========================================================
    print("\nRetraining best model on full training set ...")

    final_model = lgb.LGBMClassifier(
        **best.params,
        scale_pos_weight=scale_pos,
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    final_model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(MODELS_DIR, "lightgbm_optuna_best.joblib")
    joblib.dump(final_model, model_path, compress=3)
    print(f"Saved -> {model_path}")

    # Save study
    study_path = os.path.join(MODELS_DIR, "optuna_study.joblib")
    joblib.dump(study, study_path)
    print(f"Saved -> {study_path}")

    print("\nOptuna tuning complete.")


if __name__ == "__main__":
    main()