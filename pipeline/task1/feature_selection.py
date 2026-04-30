"""
Task 1 — Feature Selection (539 → 310 features)
Input:  data/National_task1_model_ready.parquet  (539 features)
Output: data/National_task1_310features.parquet  (310 features)

Steps:
  1. Drop AssessmentYear
  2. Variance threshold: drop columns where ≥99% of values are identical
  3. Correlation filter: drop one of each pair with |r| > 0.95
  4. Fill NaN → 0
  5. Target correlation filter: drop columns with |r| < 0.01
"""

import numpy as np
import pandas as pd

print("Loading data...")
df = pd.read_parquet("/home/fpd16/independentStudy/data/National_task1_model_ready.parquet")
print(f"Shape: {df.shape}")

X = df.drop(columns=["target"])
y = df["target"]

# Step 1: drop AssessmentYear
if "AssessmentYear" in X.columns:
    X = X.drop(columns=["AssessmentYear"])
print(f"After dropping AssessmentYear: {X.shape}")

# Step 2: variance threshold — drop if ≥99% of rows share the most common value
low_var = [col for col in X.columns if X[col].value_counts(normalize=True).iloc[0] >= 0.99]
X = X.drop(columns=low_var)
print(f"After variance threshold (dropped {len(low_var)}): {X.shape}")

# Step 3: correlation filter — drop one of each pair with |r| > 0.95
corr_matrix = X.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X.drop(columns=to_drop_corr)
print(f"After correlation filter (dropped {len(to_drop_corr)}): {X.shape}")

# Step 4: impute remaining NaN → 0
nan_count = X.isnull().sum().sum()
X = X.fillna(0)
print(f"Imputed {nan_count:,} NaN values with 0")

# Step 5: target correlation filter — drop if |r| < 0.01
target_corr = X.corrwith(y).abs()
low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
X = X.drop(columns=low_corr_cols)
print(f"After target correlation filter (dropped {len(low_corr_cols)}): {X.shape}")

# Save
df_final = X.copy()
df_final["target"] = y.values
df_final.to_parquet("/home/fpd16/independentStudy/data/National_task1_310features.parquet", index=False)
print(f"\nSaved: {df_final.shape}")
print(f"Target distribution:\n{df_final['target'].value_counts(normalize=True).round(3)}")
