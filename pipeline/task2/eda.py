"""
Task 2 EDA — Feature analysis and baseline models on the 330-feature dataset.
Input:  data/National_Task2_330features.parquet
Output: figures/task2/top20_correlations_postFS.png  (and console output)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
)
import lightgbm as lgb

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_330features.parquet")
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df["target"].value_counts(normalize=True).round(3))
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nDtypes:\n{df.dtypes.value_counts()}")

# ── Feature category counts ───────────────────────────────────────────────────
categories = {
    "Clinical/ADL": [c for c in df.columns if any(x in c for x in
                     ["ADL", "Current_", "Prior_", "Ambulation", "Transfer", "Bathing", "Grooming"])],
    "Comorbidity":  [c for c in df.columns if any(x in c for x in
                     ["comorbidity", "charlson", "elixhauser", "hfrs", "chf", "diab", "copd"])],
    "ICD_chapters": [c for c in df.columns if "-" in c and len(c) <= 8],
    "SDOH":         [c for c in df.columns if c.startswith("ACS_") or c.startswith("CEN_") or c.startswith("POS_")],
    "Diagnosis_flags": [c for c in df.columns if c.startswith("dx_")],
    "Risk_flags":   [c for c in df.columns if "Risk_For" in c],
}
for cat, cols in categories.items():
    print(f"{cat}: {len(cols)} features")

# ── Top-20 correlations with target ───────────────────────────────────────────
correlations = df.drop(columns=["target"]).corrwith(df["target"]).sort_values()
top20 = pd.concat([correlations.head(20), correlations.tail(20)])

fig, ax = plt.subplots(figsize=(10, 12))
colors = ["steelblue" if v < 0 else "salmon" for v in top20.values]
ax.barh(top20.index, top20.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Correlation with Target")
ax.set_title("Top 20 Correlations After Feature Selection")
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/task2_top20_correlations_postFS.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved task2_top20_correlations_postFS.png")

# ── Train / test split ────────────────────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pos = y_train.sum()
neg = (y_train == 0).sum()
scale_pos_weight = pos / neg
print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

# ── Baseline models ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
print("\n=== Logistic Regression Baseline ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_prob_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print("=== Random Forest Baseline ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_prob_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# LightGBM baseline (no weighting)
lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
print("=== LightGBM Baseline ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lgbm):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_prob_lgbm):.4f}")
print(classification_report(y_test, y_pred_lgbm))

# LightGBM is_unbalance=True
lgbm_u = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1, is_unbalance=True)
lgbm_u.fit(X_train, y_train)
y_pred_lgbm_u = lgbm_u.predict(X_test)
y_prob_lgbm_u = lgbm_u.predict_proba(X_test)[:, 1]
print("=== LightGBM is_unbalance ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lgbm_u):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_prob_lgbm_u):.4f}")
print(classification_report(y_test, y_pred_lgbm_u))
