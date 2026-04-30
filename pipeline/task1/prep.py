"""
Task 1 — Cohort Preparation (STUB)

Input:  data/National_OASIS_surface_cleaned.parquet
Output: data/National_task1_model_ready.parquet  (539 features, episode-level)

The full prep logic lives in src/Untitled Folder/EDA_Combined_db.ipynb.
That notebook was too exploratory to convert cleanly — it interleaves Task 1
and Task 2 cohort definitions, extensive EDA cells, and several dead-end
branches across 50+ cells. A clean rewrite was not attempted.

High-level steps performed by the notebook:
  1. Load National_OASIS_surface_cleaned.parquet
  2. Define Task 1 cohort: keep episodes where outcome is 'Stable' or
     'Hospitalized'; drop 'ED Only' rows.
  3. Build episode-level target: 1 = Hospitalized, 0 = Stable (any
     Hospital_Reason_* flag set across any row in the episode).
  4. Deduplicate to one row per episode using the first assessment date.
  5. Drop identifier columns (Beneficiary_ID, Episode_ID, Agency IDs, etc.).
  6. Drop leakage columns: Hospital_Reason_*, Emergent_Care_*, Inpatient_*,
     temporal fields (NumVisits, Days_Cared_For, Discharge_Disposition, etc.).
  7. Filter to SOC >= 2017-01-01.
  8. Extract Assessment_Month from Assessment_Effective_Date; drop raw dates.
  9. Drop columns with >90% null values.
 10. Engineer 12 ICD-10 chapter flags (dx_hypertension, dx_copd, etc.) from
     Primary_Diagnosis_ICD_10_C_M_Code and Other_Diagnosis_Code_1-5.
 11. Drop raw ICD text columns and zero-variance columns.
 12. Compute BMI from Height_in_inches and Weight_in_pounds; drop raw fields.
 13. Encode Gender (Female=1), fix ordinal columns ('UK'/'-'/'88' → NaN),
     one-hot encode ByDiscipline and REGION.
 14. Save to data/National_task1_model_ready.parquet  (shape ~1.8M × 539).

To regenerate: run src/Untitled Folder/EDA_Combined_db.ipynb end-to-end.
"""
