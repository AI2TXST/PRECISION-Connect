"""
Task 2 — Feature Selection (→ 330 features)
Converted from src/Untitled Folder/drop_SCORES_2.ipynb

Input:  data/National_Task2_ED_Cohort_encoded.parquet
Output: data/National_Task2_330features.parquet

Steps:
  1. Drop comorbidity score columns (charlson, elixhauser, hfrs) and AssessmentYear
  2. Variance threshold: drop columns where >=99% of values are identical
  3. Correlation filter: drop one of each pair with |r| > 0.95
  4. Fill NaN -> 0
  5. Target correlation filter: drop columns with |r| < 0.01
"""

import numpy as np
import pandas as pd

print("Loading Task 2 ED cohort...")
df2 = pd.read_parquet("/home/fpd16/independentStudy/data/National_Task2_ED_Cohort_encoded.parquet")
bool_cols = df2.select_dtypes(include=["bool"]).columns.tolist()
df2[bool_cols] = df2[bool_cols].astype(int)

# Step 1: drop comorbidity scores upfront
score_cols = ["comorbidity_score_charlson", "comorbidity_score_elixhauser", "hfrs", "AssessmentYear"]
df2 = df2.drop(columns=[c for c in score_cols if c in df2.columns], errors="ignore")
print(f"After dropping scores: {df2.shape}")

X2 = df2.drop(columns=["target"])
y2 = df2["target"]

# Step 2: variance threshold
threshold = 0.99
low_var = [col for col in X2.columns if X2[col].value_counts(normalize=True).iloc[0] >= threshold]
X2 = X2.drop(columns=low_var)
print(f"After variance threshold: {X2.shape}")

# Step 3: correlation filter
corr_matrix = X2.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X2 = X2.drop(columns=to_drop_corr)
print(f"After correlation filter: {X2.shape}")

# Step 4: impute missing
X2 = X2.fillna(0)

# Step 5: target correlation filter
target_corr = X2.corrwith(y2).abs()
low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
X2 = X2.drop(columns=low_corr_cols)
print(f"After target correlation filter: {X2.shape}")

df2_final = X2.copy()
df2_final["target"] = y2.values

out_path = "/home/fpd16/independentStudy/data/National_Task2_330features.parquet"
df2_final.to_parquet(out_path, index=False)
print(f"Saved: {out_path}  shape={df2_final.shape}")
