"""
Multiclass model: Stable / ED-Only / Hospitalized (3-class).
Source: src/EDA/04_modeling_clean.ipynb

This is the original framing that motivated the two-task reframe.
See pipeline/multiclass/README.md for context.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
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
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["Stable", "ED Only", "Hospitalized"]
RANDOM_STATE  = 42
DATA_PATH     = "/home/fpd16/independentStudy/data/datapreprocessed.csv"
FIGURES_DIR   = "/home/fpd16/independentStudy/figures"
MODELS_DIR    = "/home/fpd16/independentStudy/models"
DROP_COLS     = [
    "Facility_Internal_ID",
    "Patient_ZIP_Code",
    "Medicare_HHA_pymnt_episode_asmt_case_mix_grp_early_later_episode",
]

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLS)
    y = df["target"].values
    X = df.drop(columns=["target"])
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
    print(f"ED % — Train: {(y_train == 1).mean() * 100:.1f}% | Val: {(y_val == 1).mean() * 100:.1f}%")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Targeted SMOTE: ED Only → 100k (train split only) ────────────────────────
def apply_smote(X_train, y_train):
    print("Applying targeted SMOTE (ED-only class 1 → 100k)...")
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5, sampling_strategy={1: 100_000})
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    print("After SMOTE:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_sm == i).sum():,}")
    return X_sm, y_sm


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
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    safe   = model_name.replace(" ", "_")
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    colors = ["steelblue", "darkorange", "green"]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{safe}_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        RocCurveDisplay.from_predictions(
            y_bin[:, i], y_prob[:, i], name=f"{name} (AUC={auc:.3f})", color=color, ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"{model_name} — ROC Curves (One vs Rest)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{safe}_roc_curves.png", dpi=150)
    plt.close()

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=color, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{safe}_pr_curves.png", dpi=150)
    plt.close()

    return results_store


# ── Models ────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train, X_val, y_val):
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", early_stopping_rounds=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Best iteration: {model.best_iteration}")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
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
    print(f"Best iteration: {model.best_iteration_}")
    return model


# ── SHAP (best model) ─────────────────────────────────────────────────────────
def run_shap(model, X_test, model_name):
    print(f"Running SHAP on: {model_name} (n=5000 sample)")
    X_shap = X_test.sample(n=5000, random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    top_n    = 10
    ed_imp   = pd.Series(np.abs(shap_values[1]).mean(axis=0), index=X_shap.columns).nlargest(top_n)
    hosp_imp = pd.Series(np.abs(shap_values[2]).mean(axis=0), index=X_shap.columns).nlargest(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=ed_imp.values, y=ed_imp.index, palette="Oranges_r", ax=ax1)
    ax1.set_title("Top 10 Features\nED Only (Class 1)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value|")
    sns.barplot(x=hosp_imp.values, y=hosp_imp.index, palette="Reds_r", ax=ax2)
    ax2.set_title("Top 10 Features\nHospitalized (Class 2)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean |SHAP Value|")
    plt.suptitle("SHAP: ED vs Hospitalization Risk Factors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/SHAP_ED_vs_Hosp.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("ED-specific features (not in Hosp top 10):", set(ed_imp.index) - set(hosp_imp.index))
    print("Hosp-specific features (not in ED top 10):", set(hosp_imp.index) - set(ed_imp.index))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    results = {}
    rf    = train_random_forest(X_train_sm, y_train_sm)
    results = evaluate_model(rf, X_test, y_test, "Random Forest", results)

    xgb_model = train_xgboost(X_train_sm, y_train_sm, X_val, y_val)
    results   = evaluate_model(xgb_model, X_test, y_test, "XGBoost", results)

    lgbm_model = train_lightgbm(X_train_sm, y_train_sm, X_val, y_val)
    results    = evaluate_model(lgbm_model, X_test, y_test, "LightGBM", results)

    # Summary table
    comparison = []
    for name, res in results.items():
        row = {"Model": name, "Macro F1": round(res["macro_f1"], 4)}
        for i, cls in enumerate(CLASS_NAMES):
            row[f"F1 ({cls})"] = round(res["per_class_f1"][i], 4)
        comparison.append(row)
    comparison_df = pd.DataFrame(comparison).set_index("Model")
    print("\n", comparison_df.to_string())

    best_name  = max(results, key=lambda x: results[x]["macro_f1"])
    best_model = {"Random Forest": rf, "XGBoost": xgb_model, "LightGBM": lgbm_model}[best_name]
    print(f"\nBest model: {best_name} (Macro F1: {results[best_name]['macro_f1']:.4f})")

    run_shap(best_model, X_test, best_name)

    joblib.dump(rf,         f"{MODELS_DIR}/multiclass_random_forest.pkl")
    joblib.dump(xgb_model,  f"{MODELS_DIR}/multiclass_xgboost.pkl")
    joblib.dump(lgbm_model, f"{MODELS_DIR}/multiclass_lightgbm.pkl")
    comparison_df.to_csv(f"{MODELS_DIR}/multiclass_results_summary.csv")
    print("Models and results saved.")


if __name__ == "__main__":
    main()
