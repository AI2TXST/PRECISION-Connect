import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
import lightgbm as lgb
import joblib
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def find_threshold(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.05):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y, pred, average='macro')
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    log(f"Best threshold: {best_thresh:.2f} (macro F1: {best_f1:.4f})")
    return best_thresh

def evaluate(model, X, y, name, thresh):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= thresh).astype(int)
    log(f"\n=== {name} ===")
    log(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")
    log(f"PR-AUC:  {average_precision_score(y, y_prob):.4f}")
    log(f"Macro F1: {f1_score(y, y_pred, average='macro'):.4f}")
    print(classification_report(y, y_pred))

log("Loading 330-feature dataset...")
df = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_330features.parquet")
X = df.drop(columns=['target'])
y = df['target']
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

# identify SDOH columns
sdoh_cols = [c for c in X.columns if c.startswith('ACS_') or c.startswith('CEN_') or c.startswith('POS_') or c.startswith('HIFLD_')]
log(f"SDOH columns to drop: {len(sdoh_cols)}")
log(f"Clinical-only features: {X.shape[1] - len(sdoh_cols)}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

best_params = {
    'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 63,
    'max_depth': -1, 'min_child_samples': 100, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'n_jobs': -1, 'random_state': 42, 'verbose': -1,
}

# --- Model 1: Clinical only (no SDOH) ---
log("Training clinical-only model (no SDOH)...")
X_train_clin = X_train.drop(columns=sdoh_cols)
X_val_clin = X_val.drop(columns=sdoh_cols)
X_test_clin = X_test.drop(columns=sdoh_cols)
log(f"Clinical-only shape: {X_train_clin.shape}")

model_clin = lgb.LGBMClassifier(**best_params)
model_clin.fit(X_train_clin, y_train)
thresh_clin = find_threshold(model_clin, X_val_clin, y_val)
evaluate(model_clin, X_val_clin, y_val, "Clinical Only — Val", thresh_clin)
evaluate(model_clin, X_test_clin, y_test, "Clinical Only — Test", thresh_clin)
joblib.dump(model_clin, "/home/fpd16/independentStudy/models/task2_clinical_only.pkl")
log("Clinical-only model saved.")

# --- Model 2: SDOH only ---
log("Training SDOH-only model...")
X_train_sdoh = X_train[sdoh_cols]
X_val_sdoh = X_val[sdoh_cols]
X_test_sdoh = X_test[sdoh_cols]
log(f"SDOH-only shape: {X_train_sdoh.shape}")

model_sdoh = lgb.LGBMClassifier(**best_params)
model_sdoh.fit(X_train_sdoh, y_train)
thresh_sdoh = find_threshold(model_sdoh, X_val_sdoh, y_val)
evaluate(model_sdoh, X_val_sdoh, y_val, "SDOH Only — Val", thresh_sdoh)
evaluate(model_sdoh, X_test_sdoh, y_test, "SDOH Only — Test", thresh_sdoh)
joblib.dump(model_sdoh, "/home/fpd16/independentStudy/models/task2_sdoh_only.pkl")
log("SDOH-only model saved.")

# --- Model 3: Combined (already run, just for reference logging) ---
log("Note: Combined 330-feature model already saved as task2_330_best_params.pkl")
log("Compare these results against that model for the SDOH ablation conclusion.")
log("ALL DONE.")
