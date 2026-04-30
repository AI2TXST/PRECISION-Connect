"""
Task 1 — LightGBM Optuna Hyperparameter Tuning
Dataset: National_task1_310features.parquet (310 features, feature-selected)
Objective: Maximize ROC-AUC on validation set
Setup: 75 trials × 5-fold CV on train split, then eval on held-out val
Output: models/lgbm_optuna_310_model.txt
        models/lgbm_optuna_310_val_metrics.json
"""

import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, classification_report
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
t0 = time.time()
df = pd.read_parquet(
    "/home/fpd16/independentStudy/data/National_task1_310features.parquet"
)
print(f"Loaded in {time.time()-t0:.1f}s  |  shape: {df.shape}")

X = df.drop(columns=["target"])
y = df["target"]
print(f"Class distribution:\n{y.value_counts(normalize=True).round(3)}")

# ── 2. Train / Val / Test split (80/10/10, stratified, same seed) ──────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\nTrain: {X_train.shape} | hosp rate: {y_train.mean():.3f}")
print(f"Val:   {X_val.shape}   | hosp rate: {y_val.mean():.3f}")
print(f"Test:  {X_test.shape}  | hosp rate: {y_test.mean():.3f}")

# Save test set — never touch again until final evaluation
np.savez(
    "models/task1_310_test_set.npz",
    X_test=X_test.values, y_test=y_test.values,
    feature_names=X.columns.tolist()
)
print("\nTest set saved to models/task1_310_test_set.npz")

# ── 3. Optuna objective ───────────────────────────────────────────────────────
N_TRIALS = 75
N_FOLDS  = 5

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1200),
        "num_leaves":        trial.suggest_int("num_leaves", 50, 300),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0),
        "class_weight":      "balanced",
        "random_state":      42,
        "verbose":           -1,
        "n_jobs":            -1,
    }

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=1
    )
    return scores.mean()

# ── 4. Run Optuna ─────────────────────────────────────────────────────────────
print(f"\nStarting Optuna: {N_TRIALS} trials × {N_FOLDS}-fold CV on train set...")
t1 = time.time()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

elapsed = (time.time() - t1) / 60
print(f"\nOptuna finished in {elapsed:.1f} minutes")
print(f"Best CV ROC-AUC: {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ── 5. Retrain best model on full train set ───────────────────────────────────
print("\nRetraining best model on full train set...")
best_params = study.best_params | {
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)

# ── 6. Evaluate on validation set ────────────────────────────────────────────
y_prob = final_model.predict_proba(X_val)[:, 1]
y_pred = final_model.predict(X_val)

auc   = roc_auc_score(y_val, y_prob)
prauc = average_precision_score(y_val, y_prob)
f1    = f1_score(y_val, y_pred, average="macro")
rec   = recall_score(y_val, y_pred, pos_label=1)
prec  = precision_score(y_val, y_pred, pos_label=1)

print(f"\n=== Validation Results (310-feature dataset) ===")
print(f"ROC-AUC:    {auc:.4f}")
print(f"PR-AUC:     {prauc:.4f}")
print(f"Macro F1:   {f1:.4f}")
print(f"Recall-hosp:    {rec:.4f}")
print(f"Precision-hosp: {prec:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Stable", "Hospitalized"]))

# ── 7. Save model and metrics ─────────────────────────────────────────────────
final_model.booster_.save_model("models/lgbm_optuna_310_model.txt")
print("\nModel saved to models/lgbm_optuna_310_model.txt")

metrics = {
    "dataset": "National_task1_310features.parquet",
    "n_features": 310,
    "n_trials": N_TRIALS,
    "n_folds": N_FOLDS,
    "best_cv_auc": round(study.best_value, 4),
    "val_roc_auc": round(auc, 4),
    "val_pr_auc": round(prauc, 4),
    "val_macro_f1": round(f1, 4),
    "val_recall_hosp": round(rec, 4),
    "val_precision_hosp": round(prec, 4),
    "best_params": study.best_params,
    "runtime_minutes": round(elapsed, 1),
}
with open("models/lgbm_optuna_310_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to models/lgbm_optuna_310_val_metrics.json")
print("\nDone.")
