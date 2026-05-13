# Multiclass: Stable / ED-Only / Hospitalized

This folder contains the original 3-class prediction model trained on the national CMS OASIS dataset. The findings from this phase motivated the two-task reframe now implemented in `pipeline/task1/` and `pipeline/task2/`.

## Problem Definition

| Label | Class | Description |
|---|---|---|
| 0 | Stable | Discharged without any acute care utilization |
| 1 | ED Only | Emergency department visit, not admitted |
| 2 | Hospitalized | Admitted to hospital (inpatient) |

## Dataset

- **Source:** National CMS OASIS dataset
- **Episodes:** 938,549
- **Features:** 114
- **Class balance:** Stable 74.9% / Hospitalized 21.6% / ED-Only 3.5%
- **Split:** 70/15/15 stratified train/val/test

## Experiments

### Imbalance Strategies Tested

| Strategy | Models |
|---|---|
| No balancing | LR, LightGBM, XGBoost, MLP |
| SMOTE | LightGBM |
| is_unbalance=True | LightGBM |
| class_weight=balanced | LightGBM, XGBoost, MLP |
| Random Undersampling (3:1:1) | LightGBM |
| Random Undersampling (1:1:1) | LightGBM |
| Threshold tuning | LightGBM |
| One-vs-Rest (OvR) | LightGBM |
| TabNet (class weights) | TabNet |

### Full Results

| Model / Strategy | Macro F1 | F1 (Stable) | F1 (ED Only) | F1 (Hosp) |
|---|---|---|---|---|
| LightGBM Optuna | 0.578 | 0.860 | 0.113 | 0.761 |
| LightGBM Random Search | 0.576 | 0.850 | 0.117 | 0.761 |
| LightGBM Grid Search | 0.576 | 0.847 | 0.120 | 0.762 |
| Undersample 3:1:1 | 0.580 | 0.890 | 0.090 | 0.750 |
| Undersample 1:1:1 | 0.540 | 0.740 | 0.120 | 0.760 |
| TabNet (class weights) | 0.560 | 0.800 | 0.130 | 0.760 |
| LightGBM class_weight=balanced | 0.542 | 0.747 | 0.122 | 0.758 |
| XGBoost class_weight | 0.550 | 0.762 | 0.124 | 0.763 |
| MLP scaled + weighted | 0.540 | 0.750 | 0.120 | 0.760 |
| OvR combined | 0.510 | 0.700 | 0.110 | 0.730 |
| Threshold tuning (best) | 0.576 | — | 0.120 | — |
| LightGBM baseline (no balance) | 0.566 | 0.924 | 0.000 | 0.774 |
| LightGBM SMOTE | 0.566 | 0.924 | 0.000 | 0.774 |
| LightGBM is_unbalance=True | 0.566 | 0.924 | 0.000 | 0.774 |
| XGBoost baseline | 0.567 | 0.925 | 0.001 | 0.776 |
| MLP baseline (unweighted) | 0.570 | 0.920 | 0.000 | 0.780 |
| MLP scaled (unweighted) | 0.570 | 0.920 | 0.000 | 0.780 |
| Logistic Regression baseline | 0.542 | 0.909 | 0.000 | 0.716 |

### One-vs-Rest Binary Results

| Task | Macro F1 | Class F1 |
|---|---|---|
| Hospitalized vs Rest | 0.83 | 0.75 |
| Stable vs Rest | 0.81 | 0.90 |
| ED Only vs Rest | 0.45 | 0.11 |

## Key Finding

The ED-Only class F1 ceiling across every strategy tested is 0.11–0.13. No combination of model architecture, imbalance handling, or threshold tuning resolved the class boundary. Analysis of the data revealed that approximately 80% of ED visits resulted in inpatient admission, making ED-Only and Hospitalized indistinguishable from OASIS start-of-care features alone. This finding motivated the reframe into two focused binary tasks.
