"""
Task 2 — Cohort Preparation
Converted from src/Untitled Folder/Prep_NA.ipynb (Task 2 branch)

Input:  data/National_OASIS_surface_cleaned.parquet
Output: data/National_Task2_ED_Cohort_encoded.parquet

Task 2 cohort: episodes where the patient visited an ED during the care
episode (any Emergent_Care_Reason_* flag set).
Target: 1 = subsequently admitted to hospital (any Hospital_Reason_* flag),
        0 = discharged from ED without admission.
"""

import numpy as np
import pandas as pd

print("Loading surface-cleaned OASIS data...")
df = pd.read_parquet("/home/fpd16/independentStudy/data/National_OASIS_surface_cleaned.parquet")
print(f"Shape: {df.shape}")

# ── 1. Define ED presenter cohort ─────────────────────────────────────────────
emergent_cols  = [c for c in df.columns if c.startswith("Emergent_Care_Reason")]
hospital_cols  = [c for c in df.columns if c.startswith("Hospital_Reason")]
inpatient_cols = [c for c in df.columns if c.startswith("Inpatient_")]

had_ed       = df[emergent_cols].any(axis=1)
had_hospital = df[hospital_cols].any(axis=1)

print(f"\nED presenters:  {had_ed.sum():,}")
print(f"ED + admitted:  {(had_ed & had_hospital).sum():,}")
print(f"ED + discharged:{(had_ed & ~had_hospital).sum():,}")

df_task2 = df[had_ed].copy()
df_task2["target"] = had_hospital[had_ed].astype(int).values
print(f"\nTask 2 shape after ED filter: {df_task2.shape}")
print(f"Admitted:   {df_task2['target'].sum():,} ({df_task2['target'].mean()*100:.1f}%)")
print(f"Discharged: {(df_task2['target']==0).sum():,} ({(df_task2['target']==0).mean()*100:.1f}%)")

# ── 2. Episode-level dedup: take first assessment row per episode ──────────────
df_task2 = (
    df_task2
    .sort_values("Assessment_Effective_Date")
    .groupby(["Beneficiary_ID", "Start_of_Care_Date"], sort=False)
    .first()
    .reset_index()
)
print(f"\nAfter episode dedup: {df_task2.shape}")

# ── 3. Drop identifier / admin columns ────────────────────────────────────────
id_drop = [
    "Beneficiary_ID", "Episode_ID", "Agency_Medicare_Number",
    "HHA_Agency_ID", "State_ID", "Submission_Date", "Branch_State",
    "Patient_State", "Patient_Overall_Status",
]
id_drop = [c for c in id_drop if c in df_task2.columns]
df_task2 = df_task2.drop(columns=id_drop)
print(f"After dropping identifiers: {df_task2.shape}")

# ── 4. Drop leakage columns ───────────────────────────────────────────────────
temporal_leak = [
    "Date_of_Last_Home_Visit", "Last_Assessment_Date",
    "NumVisits", "Days_Cared_For", "DaysBetweenVisits", "PrevVisitDate",
    "Discharge_Disposition", "Discharge_Transfer_Death_Date",
    "Most_Recent_Inpatient_Discharge_Date", "Most_Recent_Inpat_Discharge_Date_UK",
    "Resumption_of_Care_Date", "Resumption_of_Care_Date_NA",
    "Emergent_care_use_since_most_recent_SOC_ROC",
]
intermediate = ["has_hosp", "outcome"]

leak_drop = (
    list(emergent_cols) + list(hospital_cols) + list(inpatient_cols)
    + temporal_leak + intermediate
)
leak_drop = [c for c in leak_drop if c in df_task2.columns]
df_task2 = df_task2.drop(columns=leak_drop)
print(f"After dropping leakage: {df_task2.shape}")

# ── 5. Filter to SOC >= 2017-01-01 ───────────────────────────────────────────
soc_temp = pd.to_datetime(df_task2["Start_of_Care_Date"])
df_task2 = df_task2[soc_temp >= "2017-01-01"].copy()
print(f"After SOC >= 2017 filter: {df_task2.shape}")

# ── 6. Extract month; drop raw date columns ───────────────────────────────────
df_task2["Assessment_Month"] = pd.to_datetime(
    df_task2["Assessment_Effective_Date"]
).dt.month
date_drop = ["Start_of_Care_Date", "Assessment_Effective_Date", "AssessmentYear", "YEAR"]
df_task2 = df_task2.drop(columns=[c for c in date_drop if c in df_task2.columns])
print(f"After date handling: {df_task2.shape}")

# ── 7. Drop >90% null columns ─────────────────────────────────────────────────
null_pct    = df_task2.isnull().mean()
high_null   = null_pct[null_pct > 0.90].index.tolist()
df_task2    = df_task2.drop(columns=high_null)
print(f"After dropping {len(high_null)} high-null columns: {df_task2.shape}")

# ── 8. ICD-10 chapter flags ───────────────────────────────────────────────────
icd_cols = (
    ["Primary_Diagnosis_ICD_10_C_M_Code"]
    + [f"Other_Diagnosis_Code_{i}_ICD_10_C_M" for i in range(1, 6)]
)
icd_cols = [c for c in icd_cols if c in df_task2.columns]

def has_dx(prefix_list):
    mask = pd.Series(False, index=df_task2.index)
    for col in icd_cols:
        for prefix in prefix_list:
            mask = mask | df_task2[col].astype(str).str.startswith(prefix)
    return mask.astype(int)

df_task2["dx_hypertension"]    = has_dx(["I10", "I11", "I12", "I13"])
df_task2["dx_copd"]            = has_dx(["J44", "J43"])
df_task2["dx_diabetes_t2"]     = has_dx(["E11"])
df_task2["dx_heart_failure"]   = has_dx(["I50"])
df_task2["dx_afib"]            = has_dx(["I48"])
df_task2["dx_coronary"]        = has_dx(["I25"])
df_task2["dx_stroke_sequelae"] = has_dx(["I69"])
df_task2["dx_pneumonia"]       = has_dx(["J18", "J15", "J16"])
df_task2["dx_uti"]             = has_dx(["N39.0"])
df_task2["dx_parkinsons"]      = has_dx(["G20"])
df_task2["dx_surgical_after"]  = has_dx(["Z47", "Z48"])
df_task2["dx_weakness"]        = has_dx(["M62.81", "R53.1"])

# Drop raw ICD text
icd_text_drop = [
    c for c in df_task2.columns
    if c.startswith("Regimen_Change")
    or (c.endswith("ICD_10_C_M") and "Severity" not in c)
    or c == "Primary_Diagnosis_ICD_10_C_M_Code"
]
df_task2 = df_task2.drop(columns=icd_text_drop)
print(f"After ICD engineering: {df_task2.shape}")

# ── 9. BMI from height/weight ─────────────────────────────────────────────────
if "Height_in_inches" in df_task2.columns and "Weight_in_pounds" in df_task2.columns:
    df_task2["Height_in_inches"] = pd.to_numeric(
        df_task2["Height_in_inches"].replace("-", np.nan), errors="coerce"
    )
    df_task2["Weight_in_pounds"] = pd.to_numeric(
        df_task2["Weight_in_pounds"].replace("-", np.nan), errors="coerce"
    )
    df_task2["BMI"] = (df_task2["Weight_in_pounds"] * 703) / (
        df_task2["Height_in_inches"] ** 2
    )
    df_task2["BMI"] = df_task2["BMI"].replace([np.inf, -np.inf], np.nan)
    df_task2.loc[(df_task2["BMI"] > 250) | (df_task2["BMI"] < 5), "BMI"] = np.nan
    df_task2 = df_task2.drop(columns=["Height_in_inches", "Weight_in_pounds"])

# ── 10. Encode categoricals ───────────────────────────────────────────────────
if "Gender" in df_task2.columns:
    df_task2["Gender"] = (df_task2["Gender"] == "Female").astype(int)

for col in ["Active_Diagnoses_Diabetes_Mellitus", "Active_Diagnoses_PVD_or_PAD"]:
    if col in df_task2.columns:
        df_task2[col] = pd.to_numeric(
            df_task2[col].replace("-", np.nan), errors="coerce"
        )

ordinal_cols = [
    "Ability_To_Hear", "Understanding_Of_Verbal_Content",
    "Patient_Treated_For_UTI_Last_14_Days", "Bowel_Incontinence_Frequency",
    "Frequency_Of_ADL_Or_IADL_Assistance_From_Caregiver",
    "Mobility_Discharge_Goal_Lying_to_Sitting",
    "Mobility_SOC_ROC_Performance_Lying_to_Sitting",
    "Drug_regimen_review", "Medication_intervention",
]
for c in ordinal_cols:
    if c in df_task2.columns:
        df_task2[c] = pd.to_numeric(
            df_task2[c].replace(["UK", "-", "88", "9"], np.nan), errors="coerce"
        )

cat_cols = [c for c in ["ByDiscipline", "REGION"] if c in df_task2.columns]
if cat_cols:
    df_task2 = pd.get_dummies(df_task2, columns=cat_cols, drop_first=False)

# ── 11. Drop zero-variance and bool-cast ──────────────────────────────────────
zero_var = ["V00-Y99", "diab", "diabwc", "msld", "blane", "diabc", "diabunc", "pud_elixhauser"]
zero_var = [c for c in zero_var if c in df_task2.columns]
df_task2 = df_task2.drop(columns=zero_var)

bool_cols = df_task2.select_dtypes(include=["bool"]).columns.tolist()
df_task2[bool_cols] = df_task2[bool_cols].astype(int)

print(f"\nFinal shape: {df_task2.shape}")
print(f"Admitted:   {df_task2['target'].sum():,} ({df_task2['target'].mean()*100:.1f}%)")
print(f"Discharged: {(df_task2['target']==0).sum():,} ({(df_task2['target']==0).mean()*100:.1f}%)")

# ── 12. Save ──────────────────────────────────────────────────────────────────
out_path = "/home/fpd16/independentStudy/data/National_Task2_ED_Cohort_encoded.parquet"
df_task2.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}")
