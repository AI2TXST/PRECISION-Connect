"""
Task 2 baselines: Logistic Regression, Random Forest, LightGBM (default and is_unbalance).
Source: src/Untitled Folder/EDA_Task2_330.ipynb (baseline section)

Task 2: among ED presenters, predict admission vs. discharge.
Binary target — 0: discharged from ED, 1: admitted (hospitalized) (~82.8% positive rate).
Dataset: National_Task2_330features.parquet (926,703 rows, 330 features)

Note: target=1 is the majority class here (admitted), target=0 is the minority (discharged).
Macro F1 / recall on class 0 (discharged) is the harder metric to optimize.
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
DATA_PATH    = "/home/fpd16/independentStudy/data/National_Task2_330features.parquet"


# ── Load & split ──────────────────────────────────────────────────────────────
def load_and_split():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    X = df.drop(columns=["target"])
    y = df["target"]

    # 80/20 split (matches notebook; train_optuna.py uses 80/10/10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {X_train.shape}  pos rate: {y_train.mean():.3f}")
    print(f"Test:  {X_test.shape}  pos rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def report(name, y_true, y_pred, y_prob):
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"AUPRC:   {average_precision_score(y_true, y_prob):.4f}")
    print(classification_report(y_true, y_pred))


# ── Baseline 1: Logistic Regression ──────────────────────────────────────────
def run_logistic_regression(X_train, X_test, y_train, y_test):
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)

    pred = lr.predict(X_test_scaled)
    prob = lr.predict_proba(X_test_scaled)[:, 1]
    report("Logistic Regression Baseline", y_test, pred, prob)
    return lr, scaler


# ── Baseline 2: Random Forest ─────────────────────────────────────────────────
def run_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)
    prob = rf.predict_proba(X_test)[:, 1]
    report("Random Forest Baseline", y_test, pred, prob)
    return rf


# ── Baseline 3: LightGBM (default) ───────────────────────────────────────────
def run_lightgbm_default(X_train, X_test, y_train, y_test):
    model = lgb.LGBMClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    report("LightGBM Baseline", y_test, pred, prob)
    return model


# ── Baseline 4: LightGBM is_unbalance=True ───────────────────────────────────
# is_unbalance internally re-weights samples so the minority class (discharged,
# class 0) gets higher weight — better macro F1 than scale_pos_weight for this task.
def run_lightgbm_is_unbalance(X_train, X_test, y_train, y_test):
    model = lgb.LGBMClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1, is_unbalance=True,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    report("LightGBM is_unbalance=True", y_test, pred, prob)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test = load_and_split()

    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)
    run_lightgbm_default(X_train, X_test, y_train, y_test)
    run_lightgbm_is_unbalance(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
