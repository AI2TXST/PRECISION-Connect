# Multiclass: Stable / ED-Only / Hospitalized

This folder contains the **original 3-class model** that preceded the two-task reframe
now used in `pipeline/task1/` and `pipeline/task2/`. It is preserved for provenance
and paper reproducibility — it is not the current production model.

## What this is

A single multiclass classifier trained on Texas OASIS home health episodes to predict
one of three outcomes at episode close:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Stable | Discharged without any acute care utilization |
| 1 | ED Only | Emergency department visit, not admitted |
| 2 | Hospitalized | Admitted to hospital (inpatient) |

## Dataset

- **File:** `data/datapreprocessed.csv`
- **Rows:** 938,549 episodes
- **Features:** 114 (after dropping 3 leaky/administrative columns)
- **Class balance:** Stable 79.9% / Hospitalized 17.1% / ED-Only 3.0%
- **Split:** 70% train / 15% val / 15% test (stratified)

## Results

| Model | Macro F1 | F1 (Stable) | F1 (ED Only) | F1 (Hospitalized) |
|-------|----------|-------------|--------------|-------------------|
| Random Forest | 0.6475 | 0.9773 | 0.0009 | 0.9642 |
| XGBoost | 0.6481 | 0.9774 | 0.0032 | 0.9638 |
| **LightGBM** | **0.6486** | **0.9774** | **0.0046** | **0.9638** |

Best model: **LightGBM, Macro F1 = 0.6486** (with targeted SMOTE: ED-Only oversampled to 100k).

## Why we moved away from this framing

Two structural problems made the 3-class framing unsuitable for the paper:

1. **ED-Only F1 is near zero regardless of balancing strategy.** Even with targeted SMOTE
   boosting the ED-Only class from 19k → 100k training samples, the best F1 for that class
   was 0.005. `baselines.py` shows that no-balancing, `is_unbalance`, and random undersampling
   all produce similarly near-zero ED-Only F1.

2. **The class boundary is incoherent.** Among patients who visited the ED, roughly 80%
   were subsequently admitted (hospitalized). The distinction between "ED Only" and
   "Hospitalized" is therefore driven more by ED admission policy at the receiving
   hospital than by anything in the patient's home health record. The classes are not
   separable from OASIS features alone.

The reframe splits the prediction into two distinct, tractable binary tasks:
- **Task 1** (`pipeline/task1/`): Among all home health patients, predict unplanned hospitalization (Stable vs. Hospitalized). n=5.5M national.
- **Task 2** (`pipeline/task2/`): Among ED presenters only, predict admission vs. discharge. n=926k national.

## Files

| File | Description |
|------|-------------|
| `train.py` | Full pipeline: load → SMOTE → RF/XGBoost/LightGBM → eval → SHAP → save |
| `baselines.py` | Ablation: no-balancing vs. is_unbalance vs. random undersampling across all three models |
