import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Try imports for optional libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# --- Data Balancing ---
def balance_data(X, y, random_state=42):
    """
    Balances the dataset using RandomUnderSampler (if majority > minority) 
    or RandomOverSampler (if minority > majority - unlikely for rare events, usually we undersample negs).
    Actually, usually for medical data (rare events), we want to undersample the majority class (negatives)
    or oversample the minority (positives).
    
    The logic from models.py was:
    If class 0 > class 1: Undersample (rus)
    Else: Oversample (ros)
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imblearn is not installed. Please install imbalanced-learn.")
        
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2:
        return X, y # Cannot balance
    
    # Assuming standard binary: 0 vs 1
    # Check simple majority counts
    # Getting counts safely
    c0 = sum(y == 0)
    c1 = sum(y == 1)
    
    if c0 > c1:
        # Majority is 0 (negative), Undersample 0s
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X, y)
    else:
        # Majority is 1 (positive), Oversample 0s? Or just Oversample 1s if they were minority?
        # If c1 >= c0, then 1 is majority or equal.
        # If we want to balance, and 0 is minority, we Oversample 0.
        ros = RandomOverSampler(random_state=random_state)
        X_res, y_res = ros.fit_resample(X, y)
        
    return X_res, y_res

# --- Result Logging ---
def save_model_results(metrics, filename="modeling_results.csv", output_dir="results"):
    """
    Appends metrics dict to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    df = pd.DataFrame([metrics])
    
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, mode='w', header=True, index=False)
    
    return filepath

# --- KAN Model Classes (PyTorch) ---
if TORCH_AVAILABLE:
    class NumpyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class UnivariateFunction(nn.Module):
        def __init__(self, hidden_units=20, dropout_p=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_units, 1)
            )

        def forward(self, x):
            return self.net(x)

    class KAN(nn.Module):
        def __init__(self, input_dim, hidden_units=20, dropout_p=0.1):
            super().__init__()
            self.d = input_dim
            self.univariates = nn.ModuleList([
                UnivariateFunction(hidden_units, dropout_p) for _ in range(2 * self.d + 1)
            ])
            self.coeffs = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim)) for _ in range(2 * self.d + 1)
            ])
            self.outer_weights = nn.Parameter(torch.randn(2 * self.d + 1))
            self.outer_bias = nn.Parameter(torch.randn(1))

        def forward(self, x):
            outputs = []
            for i in range(2 * self.d + 1):
                comb = x @ self.coeffs[i]
                comb = comb.unsqueeze(1)
                out = self.univariates[i](comb)
                outputs.append(out.squeeze(1))
            stacked = torch.stack(outputs, dim=1)
            return (stacked @ self.outer_weights + self.outer_bias).squeeze(-1)
else:
    # Dummy classes to prevent ImportErrors in main script if unused
    class KAN: pass
# --- Model Training Wrapper ---

def train_evaluate_model(model_name, X_train, y_train, X_test, y_test, params=None):
    """
    Trains and evaluates a model. Returns metrics dict and trained model.
    """
    if params is None:
        params = {}
    
    # Defaults
    n_estimators = params.get('n_estimators', 100)
    max_depth = params.get('max_depth', 5)
    learning_rate = params.get('learning_rate', 0.1)
    random_state = 42

    model = None
    
    try:
        if model_name == "LightGBM":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state, verbose=-1)
        
        elif model_name == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        
        elif model_name == "CatBoost":
            import catboost as cb
            model = cb.CatBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, depth=max_depth, random_state=random_state, verbose=0)
        
        elif model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        
        elif model_name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        
        elif model_name == "KNN (K-Nearest Neighbors)":
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = params.get('n_neighbors', 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            
        elif model_name == "Stacked Ensemble":
            from sklearn.ensemble import VotingClassifier, RandomForestClassifier
            import lightgbm as lgb
            import xgboost as xgb
            import catboost as cb
            
            estimators = [
                ('lgbm', lgb.LGBMClassifier(n_estimators=n_estimators, random_state=random_state, verbose=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=random_state)),
                ('cb', cb.CatBoostClassifier(n_estimators=n_estimators, verbose=0, random_state=random_state)),
                ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
            ]
            model = VotingClassifier(estimators=estimators, voting='soft')
            
        elif model_name == "Kolmogorov-Arnold Network (KAN)":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available for KAN")
            
            # KAN Training Logic (Simplified for Batch)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            epochs = params.get('epochs', 50)
            batch_size = params.get('batch_size', 256)
            hidden = params.get('hidden_units', 20)
            lr = params.get('learning_rate', 0.01)
            
            input_dim = X_train.shape[1]
            kan_model = KAN(input_dim=input_dim, hidden_units=hidden).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(kan_model.parameters(), lr=lr, weight_decay=1e-4)
            
            # Convert to Tensor (assuming X_train is numpy/df)
            X_tr_np = X_train.values if hasattr(X_train, 'values') else X_train
            y_tr_np = y_train.values if hasattr(y_train, 'values') else y_train
            
            train_loader = DataLoader(NumpyDataset(X_tr_np, y_tr_np), batch_size=batch_size, shuffle=True)
            
            for ep in range(epochs):
                kan_model.train()
                for X_b, y_b in train_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    out = kan_model(X_b).squeeze()
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
            
            model = kan_model
            
        # Fitting
        if model_name != "Kolmogorov-Arnold Network (KAN)":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # KAN Eval
            model.eval()
            X_te_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_te_t = torch.as_tensor(X_te_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(X_te_t).squeeze()
                y_prob = torch.sigmoid(logits).cpu().numpy()
                y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
        }
        
        return metrics, model
        
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None, None
