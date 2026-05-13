# Connect-2.0

## Publication

The findings from this repository have contributed to the following academic works:
*   **Journal Article (Under Review):** *ACM BCB 2026*.

## System Build Process

To build the Connect System, run the following notebooks in the following order. Each step processes raw data into the formats required by the system.

### 1. SDOH_Census_Tract_EDA.ipynb
*   **Description**: Explores and processes Social Determinants of Health (SDOH) census tract data.
*   **Inputs**: Raw Census/ACS Data, `texas_zip_codes_county.csv`
*   **Outputs**: `streamlit_sdoh.csv`

### 2. DashboardPrep_v2.ipynb
*   **Description**: Initial preprocessing of raw patient/OASIS data to create the base cohort dataset.
*   **Inputs**: Raw OASIS Data Files
*   **Outputs**: Base Patient Cohort CSV (e.g., `merged_data_v2.csv`)

### 3. riskScoreSystem_v2.ipynb
*   **Description**: Calculates patient risk scores (e.g., Charlson Comorbidity Index, Readmission Risk) for the cohort.
*   **Inputs**: Base Patient Cohort CSV
*   **Outputs**: `patientRiskScores.csv`

### 4. icdSectionAnalysis_v2.ipynb
*   **Description**: Analyzes and aggregates ICD codes for comorbidity visualization.
*   **Inputs**: `patientRiskScores.csv` or Cohort Data
*   **Outputs**: `dashboardData_ICDSections_v2.csv`

### 5. finalDashboardData_v2.ipynb
*   **Description**: Consolidates all processed data into the final dataframe used by the application.
*   **Inputs**: `patientRiskScores.csv`, `dashboardData_ICDSections_v2.csv`, `streamlit_sdoh.csv`
*   **Outputs**: `streamlit_df.csv`

---

## How to Run the App

The main app script is `County_Comorbidity_Analysis.py`.


### App Inputs
The application requires the following files in the `data` or parent directories 
(relative paths vary, ensure consistency):
*   `streamlit_df.csv` (Main Patient Data)
*   `texas_counties.geojson` (Map Geometries)
*   `streamlit_sdoh.csv` (SDOH Data)
*   `dashboardData_ICDSections_v2.csv` (ICD Section Analysis)

### Command
To launch the dashboard, use the following commands:
```bash
cd "Connect-2.0"
streamlit run County_Comorbidity_Analysis.py
```
```

```bibtex
@inproceedings{elizondo2026acmbcb,
  author    = {Elizondo, Mirna},
  title     = {PRECISION-Connect: AI-Ready Multimorbidity and SDOH Risk Vectors for Explainable 30-Day Readmission and County-Level Disparity Modeling},
  booktitle = {Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (ACM BCB)},
  year      = {2026},
  note      = {Poster}
}
