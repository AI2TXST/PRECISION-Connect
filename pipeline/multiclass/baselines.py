"""
Multiclass baseline comparison: no-balancing vs. is_unbalance vs. random undersampling.
Source: src/EDA/05_modeling_baseline.ipynb

Establishes baselines for the 3-class problem (Stable / ED-Only / Hospitalized)
to show that ED-Only F1 is near 0 without SMOTE — motivating train.py's targeted SMOTE.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    label_binarize,
    precision_recall_curve,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Stable", "ED Only", "Hospitalized"]
RANDOM_STATE = 42
DATA_PATH    = "/home/fpd16/independentStudy/data/datapreprocessed.csv"
FIGURES_DIR  = "/home/fpd16/independentStudy/figures/baseline"
DROP_COLS    = [
    "Facility_Internal_ID",
    "Patient_ZIP_Code",
    "Medicare_HHA_pymnt_episode_asmt_case_mix_grp_early_later_episode",
]

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLS)
    y  = df["target"].values
    X  = df.drop(columns=["target"])
    print(f"Shape: {X.shape}")
    print("Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} ({name}): {(y == i).sum():,} ({(y == i).mean() * 100:.1f}%)")
    return X, y


# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print("No balancing applied — raw class distribution.")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name, results_store):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    macro_f1     = f1_score(y_test, y_pred, average="macro")
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {per_class_f1[i]:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

    results_store[model_name] = {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }

    safe  = model_name.replace(" ", "_")
    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{safe}_confusion_matrix.png", dpi=150)
    plt.close()

    return results_store


# ── Experiment 1: No balancing ────────────────────────────────────────────────
def run_no_balancing(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n=== No Balancing ===")
    results = {}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results = evaluate_model(rf, X_test, y_test, "RF No Balancing", results)

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", early_stopping_rounds=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    results = evaluate_model(xgb_model, X_test, y_test, "XGBoost No Balancing", results)

    lgbm_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    results = evaluate_model(lgbm_model, X_test, y_test, "LightGBM No Balancing", results)

    return results


# ── Experiment 2: LightGBM is_unbalance=True ─────────────────────────────────
def run_is_unbalance(X_train, X_val, y_train, y_val, X_test, y_test, results):
    print("\n=== LightGBM is_unbalance=True ===")
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    return evaluate_model(model, X_test, y_test, "LightGBM is_unbalance=True", results)


# ── Experiment 3: Random undersampling ───────────────────────────────────────
def run_random_undersampling(X_train, X_val, X_test, y_train, y_val, y_test, results):
    print("\n=== Random Undersampling ===")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    print("After Random Undersampling:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_rus == i).sum():,}")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_rus, y_rus)
    results = evaluate_model(rf, X_test, y_test, "RF Random Undersampling", results)

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", early_stopping_rounds=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_rus, y_rus, eval_set=[(X_val, y_val)], verbose=50)
    results = evaluate_model(xgb_model, X_test, y_test, "XGBoost Random Undersampling", results)

    lgbm_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    lgbm_model.fit(
        X_rus, y_rus,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    results = evaluate_model(lgbm_model, X_test, y_test, "LightGBM Random Undersampling", results)

    return results


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results):
    rows = []
    for name, res in results.items():
        row = {"Model": name, "Macro F1": round(res["macro_f1"], 4)}
        for i, cls in enumerate(CLASS_NAMES):
            row[f"F1 ({cls})"] = round(res["per_class_f1"][i], 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    print("\nBaseline Results:")
    print(df.to_string())
    df.to_csv("/home/fpd16/independentStudy/multiclass_baseline_results.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    results = run_no_balancing(X_train, X_val, X_test, y_train, y_val, y_test)
    results = run_is_unbalance(X_train, X_val, y_train, y_val, X_test, y_test, results)
    results = run_random_undersampling(X_train, X_val, X_test, y_train, y_val, y_test, results)

    print_summary(results)


if __name__ == "__main__":
    main()
