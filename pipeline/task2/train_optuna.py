import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score
import lightgbm as lgb
import optuna
import joblib
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading 330-feature dataset...")
df = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_330features.parquet")
X = df.drop(columns=['target'])
y = df['target']
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
log(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

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

def evaluate(model, X, y, name, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y, y_prob)
    log(f"\n=== {name} ===")
    log(f"AUC: {auc:.4f}")
    print(classification_report(y, y_pred))
    return auc

param_space = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [31, 63, 127],
    'max_depth': [5, 10, -1],
    'min_child_samples': [50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0.1, 1.0],
}

# Optuna + class_weight
log("Starting Optuna + class_weight=balanced...")
t0 = time.time()
def objective_cw(trial):
    params = {k: trial.suggest_categorical(k, v) for k, v in param_space.items()}
    model = lgb.LGBMClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log(f"  Trial {trial.number}: AUC={auc:.4f} | params={trial.params}")
    return auc

study_cw = optuna.create_study(direction='maximize')
study_cw.optimize(objective_cw, n_trials=20)
log(f"Optuna (class_weight) done in {(time.time()-t0)/60:.1f} min")
log(f"Best params: {study_cw.best_params}")
best_cw = lgb.LGBMClassifier(**study_cw.best_params, class_weight='balanced', n_jobs=-1, random_state=42, verbose=-1)
best_cw.fit(X_train, y_train)
evaluate(best_cw, X_val, y_val, "Optuna + class_weight (330 features)")
joblib.dump(best_cw, "/home/fpd16/independentStudy/models/task2_330_optuna_cw.pkl")
log("Model saved: task2_330_optuna_cw.pkl")

# Optuna + threshold
log("Starting Optuna + threshold tuning...")
t0 = time.time()
def objective_th(trial):
    params = {k: trial.suggest_categorical(k, v) for k, v in param_space.items()}
    model = lgb.LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log(f"  Trial {trial.number}: AUC={auc:.4f} | params={trial.params}")
    return auc

study_th = optuna.create_study(direction='maximize')
study_th.optimize(objective_th, n_trials=20)
log(f"Optuna (threshold) done in {(time.time()-t0)/60:.1f} min")
log(f"Best params: {study_th.best_params}")
best_th = lgb.LGBMClassifier(**study_th.best_params, n_jobs=-1, random_state=42, verbose=-1)
best_th.fit(X_train, y_train)
thresh = find_threshold(best_th, X_val, y_val)
evaluate(best_th, X_val, y_val, "Optuna + threshold (330 features)", threshold=thresh)
joblib.dump(best_th, "/home/fpd16/independentStudy/models/task2_330_optuna_th.pkl")
log("Model saved: task2_330_optuna_th.pkl")

log("ALL DONE.")
