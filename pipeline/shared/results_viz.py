"""
Results Visualization — Final test metrics and model comparison lollipop plots.
Hardcoded final test results for Task 1 and Task 2, plus all model comparison plots.
Outputs:
  figures/task1_lollipop.png
  figures/task2_lollipop.png
  figures/task2_fs_lollipop.png
  figures/final_cm_task1.png
  figures/final_cm_task2.png
"""

import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    classification_report, average_precision_score,
)

# ── Final test results ────────────────────────────────────────────────────────
test1 = np.load("models/task1_310_test_set.npz", allow_pickle=True)
X_test1 = test1["X_test"]
y_test1 = test1["y_test"]

model1 = lgb.Booster(model_file="models/lgbm_optuna_310_model.txt")
y_prob1 = model1.predict(X_test1)
y_pred1 = (y_prob1 >= 0.5).astype(int)

print("=== TASK 1 FINAL TEST RESULTS ===")
print(f"ROC-AUC: {roc_auc_score(y_test1, y_prob1):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test1, y_prob1):.4f}")
print(f"Macro F1: {f1_score(y_test1, y_pred1, average='macro'):.4f}")
print(classification_report(y_test1, y_pred1, target_names=["Stable", "Hospitalized"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test1, y_pred1))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

model2 = joblib.load("/home/fpd16/independentStudy/models/task2_330_optuna_cw.pkl")
df2 = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_330features.parquet")
X2 = df2.drop(columns=["target"])
y2 = df2["target"]
bool_cols = X2.select_dtypes(include=["bool"]).columns.tolist()
X2[bool_cols] = X2[bool_cols].astype(int)
_, X2_temp, _, y2_temp = train_test_split(X2, y2, test_size=0.2, stratify=y2, random_state=42)
_, X2_test, _, y2_test = train_test_split(X2_temp, y2_temp, test_size=0.5, stratify=y2_temp, random_state=42)

print(f"\nTest set: {X2_test.shape} | admitted rate: {y2_test.mean():.3f}")
y_prob2 = model2.predict_proba(X2_test)[:, 1]
y_pred2 = model2.predict(X2_test)

print("\n=== TASK 2 FINAL TEST RESULTS ===")
print(f"ROC-AUC: {roc_auc_score(y2_test, y_prob2):.4f}")
print(f"Macro F1: {f1_score(y2_test, y_pred2, average='macro'):.4f}")
print(classification_report(y2_test, y_pred2, target_names=["Discharged", "Admitted"]))
print(confusion_matrix(y2_test, y_pred2))

# ── Confusion matrix plots ────────────────────────────────────────────────────
def plot_cm(cm, class_names, title, auc, f1, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{title}\nROC-AUC: {auc:.4f}  |  Macro F1: {f1:.4f}", fontsize=12, pad=15)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i,j]:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")

cm1 = np.array([[356545, 91482], [20449, 80362]])
plot_cm(cm1, ["0: Stable", "1: Hospitalized"],
        "Task 1 — Hospitalized vs Stable (Test Set)",
        0.8838, 0.7269, "final_cm_task1.png")

cm2 = np.array([[11268, 4707], [20580, 56116]])
plot_cm(cm2, ["0: Discharged", "1: Admitted"],
        "Task 2 — ED Admitted vs Discharged (Test Set)",
        0.7967, 0.6437, "final_cm_task2.png")

# ── Task 1 lollipop: model comparison ────────────────────────────────────────
models_t1 = [
    "Logistic Regression",
    "Random Forest",
    "MLP",
    "TabNet",
    "TabPFN (10K)",
    "XGBoost RS",
    "LightGBM Baseline",
    "LightGBM Optuna",
    "XGBoost",
    "LightGBM RS",
]
f1_t1 = [0.432, 0.664, 0.686, 0.689, 0.690, 0.695, 0.728, 0.736, 0.730, 0.743]

sorted_t1 = sorted(zip(f1_t1, models_t1))
f1_t1, models_t1 = zip(*sorted_t1)
colors_t1 = ["#0A9396" if m == "LightGBM RS" else "#B0C4D4" for m in models_t1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hlines(y=models_t1, xmin=0.3, xmax=f1_t1, color="#CCCCCC", linewidth=1.5)
ax.scatter(f1_t1, models_t1, color=colors_t1, s=120, zorder=5)
for f1, model in zip(f1_t1, models_t1):
    ax.text(f1 + 0.005, model, f"{f1:.3f}", va="center", fontsize=9)
ax.set_xlabel("Macro F1 Score", fontsize=11)
ax.set_title("Task 1 — Model Comparison (Macro F1)\nHospitalized vs Stable",
             fontsize=13, fontweight="bold")
ax.set_xlim(0.3, 0.82)
ax.axvline(x=0.743, color="#0A9396", linestyle="--", linewidth=1, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/task1_lollipop.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved task1_lollipop.png")

# ── Task 2 lollipop: full-feature model comparison ────────────────────────────
models_t2 = [
    "Logistic Regression\n(none)",
    "Random Forest\n(none)",
    "LightGBM\n(none)",
    "XGBoost\n(none)",
    "LightGBM\n(class_weight=balanced)",
    "LightGBM\n(is_unbalance=True)",
    "LightGBM\n(threshold=0.70)",
    "LightGBM\n(undersample majority)",
    "LightGBM Optuna\n(threshold=0.70)",
]
f1_t2 = [0.45, 0.52, 0.60, 0.62, 0.64, 0.64, 0.69, 0.69, 0.70]

sorted_t2 = sorted(zip(f1_t2, models_t2))
f1_t2, models_t2 = zip(*sorted_t2)
colors_t2 = ["#0A9396" if "Optuna" in m else "#B0C4D4" for m in models_t2]

fig, ax = plt.subplots(figsize=(10, 7))
ax.hlines(y=models_t2, xmin=0.3, xmax=f1_t2, color="#CCCCCC", linewidth=1.5)
ax.scatter(f1_t2, models_t2, color=colors_t2, s=120, zorder=5)
for f1, model in zip(f1_t2, models_t2):
    ax.text(f1 + 0.005, model, f"{f1:.2f}", va="center", fontsize=9)
ax.set_xlabel("Macro F1 Score", fontsize=11)
ax.set_title("Task 2 — Model Comparison (Macro F1)\nED Admitted vs Discharged",
             fontsize=13, fontweight="bold")
ax.set_xlim(0.3, 0.80)
ax.axvline(x=0.70, color="#0A9396", linestyle="--", linewidth=1, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/task2_lollipop.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved task2_lollipop.png")

# ── Task 2 lollipop: feature-selected model comparison ────────────────────────
models_t2_fs = [
    "LightGBM\n(scale_pos_weight=4.80)",
    "Random Forest\n(none)",
    "Random Forest\n(class_weight=balanced)",
    "Logistic Regression\n(none)",
    "LightGBM\n(none)",
    "Logistic Regression\n(class_weight=balanced)",
    "LightGBM\n(is_unbalance=True)",
    "LightGBM Optuna\n(class_weight=balanced)",
    "LightGBM Optuna\n(threshold=0.70)",
]
f1_t2_fs = [0.48, 0.53, 0.53, 0.54, 0.60, 0.60, 0.63, 0.68, 0.683]

sorted_fs = sorted(zip(f1_t2_fs, models_t2_fs))
f1_t2_fs, models_t2_fs = zip(*sorted_fs)
colors_fs = ["#0A9396" if ("Optuna" in m and "threshold" in m) else "#B0C4D4" for m in models_t2_fs]

fig, ax = plt.subplots(figsize=(10, 7))
ax.hlines(y=models_t2_fs, xmin=0.3, xmax=f1_t2_fs, color="#CCCCCC", linewidth=1.5)
ax.scatter(f1_t2_fs, models_t2_fs, color=colors_fs, s=120, zorder=5)
for f1, model in zip(f1_t2_fs, models_t2_fs):
    ax.text(f1 + 0.005, model, f"{f1:.3f}", va="center", fontsize=9)
ax.set_xlabel("Macro F1 Score", fontsize=11)
ax.set_title("Task 2 — Feature Selection Model Comparison (Macro F1)\n"
             "ED Admitted vs Discharged · 330 features",
             fontsize=13, fontweight="bold")
ax.set_xlim(0.3, 0.78)
ax.axvline(x=0.683, color="#0A9396", linestyle="--", linewidth=1, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/task2_fs_lollipop.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved task2_fs_lollipop.png")
