# Project Proposal Title  
*Extending CONNECT: A Comprehensive Framework for Feature Selection and Ensemble Modeling in High-Dimensional EHR Data*

---

## Teammates
- Names and NetIDs

---

## Project Abstract
This project extends the existing CONNECT framework to develop a more comprehensive and scalable machine learning pipeline for modeling high-dimensional Electronic Health Records (EHR) data. We aim to integrate additional feature selection methods—such as variance thresholds, model-based selection, permutation importance, and sequential feature selectors—alongside ensemble learning algorithms including Random Forest, XGBoost, LightGBM, CatBoost, and Gradient Boosting Machines.

By evaluating these approaches on large-scale clinical datasets, specifically the CMS OASIS dataset, this project seeks to improve predictive accuracy, model robustness, and interpretability. The extended framework will support reproducible experiments, benchmarking, and systematic comparison across feature selection techniques and ensemble models.

---

## Problem Statement

### Why the Problem Is Important
Healthcare systems generate vast amounts of clinical data, yet effectively identifying informative predictors from high-dimensional EHR datasets remains a major challenge. Traditional feature selection and baseline models used in the CONNECT framework do not fully exploit modern ensemble learning techniques or more sophisticated selection strategies. This gap limits model performance, interpretability, and clinical usefulness.

### Clear Statement of the Problem
Current EHR predictive modeling pipelines often underperform because:
1. High-dimensional clinical data require careful feature selection to avoid noise and multicollinearity.  
2. Baseline models lack the capacity to capture complex, nonlinear relationships in heterogeneous patient data.

We propose to extend the CONNECT framework to incorporate a more diverse set of feature selection methods and robust ensemble models to deliver stronger and more stable predictions.

### Benchmark to Use and Why
The benchmark will be the baseline CONNECT models, which include a LightGBM implementation.

The CONNECT system is built around three core components:  
1. Aggregating diverse datasets (CMS, EHR, SDOH, etc.), as reflected in the “datasetQuickView.ipynb” notebook.  
2. Implementing Comorbidity scores, HFRS, Disability and Impairment Scores, and calculating disparity metrics (Absolute/Relative Index Disparity, Population-Weighted Disparity, Attributable Disparity).  
3. Developing an intuitive decision-support system tailored for healthcare practitioners.

---

## Dataset

The system aggregates structured and unstructured data from various public health sources, including:

- **CMS OASIS Dataset** – Home health data for comorbidity and outcome modeling.  
- **OpenDataSoft Georef U.S. ZIP Code Point Dataset** – ZIP-code–level geographic information.  
- **2020 Census Urban Area to County Dataset** – Demographic data for population and urban/rural classification.  
- **Social Determinants of Health (SDOH) Data** – Income, education, employment, and other social factors.

### Table 1. Dataset Overview and Utilization

| Name | Rows | Columns | Columns Used |
|------|------|---------|---------------|
| CMS OASIS | 15,754,330 | 271 | 27 |
| OpenDataSoft Georef | 2,661 | 15 | 4 |
| 2020 Census UA | 3,234 | 33 | 10 |
| SDOH | 85,529 | 329 | 70 |

---

## Methodology

### Baseline Tools
1. Core CONNECT system components (county selection, demographics, choropleth maps).  
2. Advanced analytics (heatmaps, treemaps, dendrograms).  
3. Initial ML integration: LightGBM classifier with real-time preprocessing, classification reports, and feature importance display.

### Feature Selection Methods Added

**Filter Methods**  
- Variance Threshold  

**Embedded Methods**  
- L1-Regularized Logistic Regression (LASSO)  
- Random Forest Feature Importance  
- Gradient Boosting Feature Importance 
- XGBoost Feature Importance
- LightGBM Feature Importance
- CatBoost Feature Importance


**Wrapper Methods**  
- Recursive Feature Elimination (RFE) with Random Forest  
- RFE with Ridge Regression  
- Permutation Importance (Random Forest)  
- Permutation Importance (Ridge Regression)  
- Sequential Feature Selection (SFS) with KNN  
- SFS with Ridge Regression  

### Models Added
- Support Vector Machine (SVM)
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- Gradient Boosting Classifier  

---

## Teaming Strategy
- Define clear roles for each member (data cleaning, model development, evaluation, visualization).  
- Meet 2–3 times per week with asynchronous coordination via Slack or Teams.  
- Use GitHub Projects and Issues for version control and task tracking.

---

## Role Assignments and Commitment Matrix
- Assign responsibilities for dataset preprocessing, feature selection pipelines, model training, results analysis, and documentation.  
- Collaboration tools: GitHub, Slack/Teams, Google Drive/OneDrive.  
- Weekly milestones ensure consistent progress.

---

## Mitigation Plan
- **If ensemble models underperform:** revert to simpler penalized models (ElasticNet, Ridge).  
- **If feature selection methods fail:** use dimensionality reduction (PCA, UMAP).  
- **If scope must shrink:** focus on core methods (e.g., 3 feature selection methods + 2 ensemble models).  
- **If baseline is GIGO:**  
  - Reevaluate data cleaning and encoding  
  - Reassess missingness strategy  
  - Apply stronger preprocessing (imputation, outlier filtering, log transforms, standardization)  

---
