"""
Task 1 baselines: Logistic Regression, Random Forest, LightGBM (default and balanced).
Source: src/Untitled Folder/Modelling_Task1_10.ipynb (baseline model cells)

Task 1: predict unplanned hospitalization among all home health patients.
Binary target — 0: Stable, 1: Hospitalized (~18.4% positive rate).
Dataset: National_task1_310features.parquet (5,488,376 rows, 310 features)

Feature selection (539 → 310 features) is handled separately in feature_selection.py.
This script loads the already-reduced 310-feature dataset.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_PATH    = "/home/fpd16/independentStudy/data/National_task1_310features.parquet"


# ── Load & split ──────────────────────────────────────────────────────────────
def load_and_split():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded: {df.shape}")
    print(f"Target:\n{df['target'].value_counts(normalize=True).round(3)}")

    X = df.drop(columns=["target"])
    y = df["target"]

    # 80/10/10 split (matches train_optuna.py)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape}  pos: {y_train.mean():.3f}")
    print(f"Val:   {X_val.shape}  pos: {y_val.mean():.3f}")
    print(f"Test:  {X_test.shape}  pos: {y_test.mean():.3f}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def report(name, y_true, y_pred, y_prob):
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_prob):.4f}")
    print(f"F1:      {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))


# ── Baseline 1: Logistic Regression ──────────────────────────────────────────
def run_logistic_regression(X_train, X_val, y_train, y_val):
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)

    prob = lr.predict_proba(X_val_scaled)[:, 1]
    pred = lr.predict(X_val_scaled)
    report("Logistic Regression", y_val, pred, prob)
    return lr, scaler


# ── Baseline 2: Random Forest ─────────────────────────────────────────────────
def run_random_forest(X_train, X_val, y_train, y_val):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    prob = rf.predict_proba(X_val)[:, 1]
    pred = rf.predict(X_val)
    report("Random Forest", y_val, pred, prob)
    return rf


# ── Baseline 3: LightGBM (default) ───────────────────────────────────────────
def run_lightgbm_default(X_train, X_val, y_train, y_val):
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_val)[:, 1]
    pred = model.predict(X_val)
    report("LightGBM (default)", y_val, pred, prob)
    return model


# ── Baseline 4: LightGBM + class_weight='balanced' ───────────────────────────
def run_lightgbm_balanced(X_train, X_val, y_train, y_val):
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_val)[:, 1]
    pred = model.predict(X_val)
    report("LightGBM (class_weight='balanced')", y_val, pred, prob)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    run_logistic_regression(X_train, X_val, y_train, y_val)
    run_random_forest(X_train, X_val, y_train, y_val)
    run_lightgbm_default(X_train, X_val, y_train, y_val)
    run_lightgbm_balanced(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()
