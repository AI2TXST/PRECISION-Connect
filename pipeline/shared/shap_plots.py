"""
SHAP Analysis — Task 1 and Task 2
Generates beeswarm SHAP summary plots for the best models from each task.
Inputs:
  models/task1_310_best_params.pkl
  models/task2_330_optuna_th.pkl
  data/National_task1_310features.parquet
  data/National_Task2_330features.parquet
Outputs:
  figures/task1_shap_summary.png
  figures/task2_shap_summary.png
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── Task 1 ────────────────────────────────────────────────────────────────────
print("Loading Task 1 data...")
best_params = joblib.load("/home/fpd16/independentStudy/models/task1_310_best_params.pkl")
print(best_params)

df = pd.read_parquet("/home/fpd16/independentStudy/data/National_task1_310features.parquet")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print("Retraining Task 1 model...")
model1 = lgb.LGBMClassifier(
    class_weight="balanced",
    colsample_bytree=0.9723148176376152,
    learning_rate=0.1069351431185152,
    min_child_samples=167,
    n_estimators=784,
    n_jobs=-1,
    num_leaves=181,
    random_state=42,
    reg_alpha=0.4975686925706463,
    reg_lambda=0.0035032852935301018,
    subsample=0.4908668948844511,
    verbose=-1,
)
model1.fit(X_train, y_train)
print("Training done")

# Stratified sample — 3000 per class
X_shap1 = pd.concat([
    X_test[y_test == 0].sample(n=3000, random_state=42),
    X_test[y_test == 1].sample(n=3000, random_state=42),
])

print("Computing SHAP values for Task 1...")
explainer1 = shap.TreeExplainer(model1)
shap_values1 = explainer1.shap_values(X_shap1)
print(type(shap_values1), np.array(shap_values1).shape)

shap.summary_plot(shap_values1, X_shap1, max_display=20, show=False)
plt.title("Task 1 — SHAP Summary (Hospitalized vs Stable)")
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/task1_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved task1_shap_summary.png")

# ── Task 2 ────────────────────────────────────────────────────────────────────
print("\nLoading Task 2 data...")
model2 = joblib.load("/home/fpd16/independentStudy/models/task2_330_optuna_th.pkl")

df2 = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_330features.parquet")
X2 = df2.drop(columns=["target"])
y2 = df2["target"]

X2_train, X2_temp, y2_train, y2_temp = train_test_split(X2, y2, test_size=0.20, stratify=y2, random_state=42)
_, X2_test, _, y2_test = train_test_split(X2_temp, y2_temp, test_size=0.50, stratify=y2_temp, random_state=42)

X_shap2 = pd.concat([
    X2_test[y2_test == 0].sample(n=3000, random_state=42),
    X2_test[y2_test == 1].sample(n=3000, random_state=42),
])

print("Computing SHAP values for Task 2...")
explainer2 = shap.TreeExplainer(model2)
shap_values2 = explainer2.shap_values(X_shap2)
print(type(shap_values2), np.array(shap_values2).shape)

shap.summary_plot(shap_values2, X_shap2, max_display=20, show=False)
plt.title("Task 2 — SHAP Summary (ED Admitted vs Discharged)")
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/task2_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved task2_shap_summary.png")
