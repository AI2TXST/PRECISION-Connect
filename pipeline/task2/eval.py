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

log("Loading Task 2 330-feature dataset...")
df = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_330features.parquet")
X = df.drop(columns=['target'])
y = df['target']
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
log(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

best_params = {
    'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 63,
    'max_depth': -1, 'min_child_samples': 100, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'n_jobs': -1, 'random_state': 42, 'verbose': -1,
}

log("Training with best Optuna params...")
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)
thresh = find_threshold(model, X_val, y_val)

for split_name, X_s, y_s in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
    y_prob = model.predict_proba(X_s)[:, 1]
    y_pred = (y_prob >= thresh).astype(int)
    log(f"\n=== Task 2 — 330 features ({split_name}) ===")
    log(f"ROC-AUC: {roc_auc_score(y_s, y_prob):.4f}")
    log(f"PR-AUC:  {average_precision_score(y_s, y_prob):.4f}")
    log(f"Macro F1: {f1_score(y_s, y_pred, average='macro'):.4f}")
    print(classification_report(y_s, y_pred))

joblib.dump(model, "/home/fpd16/independentStudy/models/task2_330_best_params.pkl")
log("Model saved. ALL DONE.")
