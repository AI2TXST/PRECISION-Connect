"""
Build National Dataset — Merge OASIS assessments with ZIP-level SDOH data
and compute episode-level comorbidity scores (Charlson, Elixhauser, HFRS).
Inputs:
  /home/m_e172/OASIS/OASIS_v2/data/OASIS_v2.csv
  /home/m_e172/OASIS/OASIS_v2/data/sdoh/SDOH_2020_ZIPCODE_1_0.csv
Output:
  data/National_OASIS_master_episode_scores.csv  (15.7M × 680)
"""

import pandas as pd
import numpy as np
from comorbidipy import comorbidity, hfrs

# ── Load SDOH ─────────────────────────────────────────────────────────────────
print("Loading SDOH data...")
sdoh_national = pd.read_csv(
    "/home/m_e172/OASIS/OASIS_v2/data/sdoh/SDOH_2020_ZIPCODE_1_0.csv",
    low_memory=False,
)
print(f"SDOH shape: {sdoh_national.shape}")
print(f"Unique ZIPs: {sdoh_national['ZIPCODE'].nunique()}")

sdoh_national["ZIP5"] = sdoh_national["ZIPCODE"].astype(str).str.zfill(5)

# ── Load OASIS ────────────────────────────────────────────────────────────────
print("\nLoading OASIS_v2.csv...")
df = pd.read_csv("/home/m_e172/OASIS/OASIS_v2/data/OASIS_v2.csv", low_memory=False)

# Normalize ZIP to 5 digits; drop rows with malformed ZIPs (< 3 digits)
df["ZIP5"] = df["Patient_ZIP_Code"].astype(str).str[:5].str.zfill(5)
df = df[df["ZIP5"].str.len() == 5]
print(f"OASIS rows: {len(df):,}")

oasis_zips = set(df["ZIP5"].unique())
sdoh_zips = set(sdoh_national["ZIP5"].unique())
print(f"OASIS ZIPs: {len(oasis_zips):,} | SDOH ZIPs: {len(sdoh_zips):,}")
print(f"Matching: {len(oasis_zips & sdoh_zips):,}")
print(f"Rows that will match: {df['ZIP5'].isin(sdoh_zips).sum():,} / {len(df):,}")

# ── Merge OASIS + SDOH ────────────────────────────────────────────────────────
print("\nMerging OASIS with SDOH...")
df_merged = df.merge(sdoh_national, left_on="ZIP5", right_on="ZIP5", how="inner")
print(f"Merged rows: {len(df_merged):,} | columns: {len(df_merged.columns):,}")
print(f"Unique patients: {df_merged['Beneficiary_ID'].nunique():,}")

# ── Build episode ID ──────────────────────────────────────────────────────────
df_merged["Episode_ID"] = (
    df_merged["Beneficiary_ID"] + "_" + df_merged["Start_of_Care_Date"].astype(str)
)

# ── Melt ICD codes for comorbidity scoring ────────────────────────────────────
diag_columns = [
    "Primary_Diagnosis_ICD_10_C_M_Code",
    "Other_Diagnosis_Code_1_ICD_10_C_M",
    "Other_Diagnosis_Code_2_ICD_10_C_M",
    "Other_Diagnosis_Code_3_ICD_10_C_M",
    "Other_Diagnosis_Code_4_ICD_10_C_M",
    "Other_Diagnosis_Code_5_ICD_10_C_M",
]

print("\nMelting ICD columns...")
df_melted = (
    df_merged[["Episode_ID"] + diag_columns]
    .melt(id_vars=["Episode_ID"], value_vars=diag_columns,
          var_name="Diagnosis_Source", value_name="ICD_Code")
    .dropna(subset=["ICD_Code"])
)
print(f"Melted rows: {len(df_melted):,} | Unique episodes: {df_melted['Episode_ID'].nunique():,}")

# ── Comorbidity scoring ───────────────────────────────────────────────────────
print("\nComputing Charlson...")
charlson_ep = comorbidity(
    df_melted.rename(columns={"Episode_ID": "BENE_ID"}),
    id="BENE_ID", code="ICD_Code", score="charlson", age=None,
)
print(f"Charlson: {charlson_ep.shape}")

print("Computing Elixhauser...")
elixhauser_ep = comorbidity(
    df_melted.rename(columns={"Episode_ID": "BENE_ID"}),
    id="BENE_ID", code="ICD_Code", score="elixhauser", age=None, weighting="vw",
)
print(f"Elixhauser: {elixhauser_ep.shape}")

print("Computing HFRS...")
hfrs_ep = hfrs(
    df_melted.rename(columns={"Episode_ID": "BENE_ID"}),
    id="BENE_ID", code="ICD_Code",
)
print(f"HFRS: {hfrs_ep.shape}")

# ── Join risk scores back ─────────────────────────────────────────────────────
risk_ep = (
    charlson_ep
    .merge(elixhauser_ep, on="BENE_ID", suffixes=("_charlson", "_elixhauser"))
    .merge(hfrs_ep, on="BENE_ID")
    .rename(columns={"BENE_ID": "Episode_ID"})
)
print(f"\nRisk scores: {risk_ep.shape}")

df_merged = df_merged.merge(risk_ep, on="Episode_ID", how="left")
score_cols = [c for c in risk_ep.columns if c != "Episode_ID"]
df_merged[score_cols] = df_merged[score_cols].fillna(0)

print(f"Final shape: {df_merged.shape}")
print(f"Unique patients: {df_merged['Beneficiary_ID'].nunique():,}")
print(f"Unique episodes: {df_merged['Episode_ID'].nunique():,}")

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "/home/fpd16/independentStudy/data/National_OASIS_master_episode_scores.csv"
df_merged.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
