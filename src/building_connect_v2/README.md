# Connect-2.0

Notebook Order:
1. SDOH_Census_Tract_EDA.ipynb
2. DashboardPrep_v2.ipynb
3. riskScoreSystem_v2.ipynb
4. icdSectionAnalysis_v2.ipynb
5. finalDashboardData_v2.ipynb



This pipeline takes raw OASIS assessments, engineers visit and ICD-based features, links social determinants of health (SDOH), and produces county-level data for Texas.

## 1. SDOH_Census_Tract_EDA.ipynb

- Purpose:
  - Explore and clean Texas SDOH and geographic reference data at ZIP, county, and tract level for later joining to patient records.
- Inputs:
  - Texas ZIP–SDOH geo-reference file (for example, texassdohtractgeorefcensus.csv) with ZIP, county name, county FIPS, population, housing units, broadband, income, poverty, and related ACS percentages.
- Outputs:
  - Cleaned SDOH dataframe indexed by ZIP and county plus derived summaries such as:
    - Number of unique ZIP codes per county.
    - Per‑county averages of SDOH percentages (education, broadband, income, poverty, etc.).
  - These outputs are later merged with patient-level data in DashboardPrep and finalDashboard notebooks.

## 2. DashboardPrep_v2.ipynb

- Purpose:
  - Transform the preprocessed OASIS data into a patient‑episode level dataset for risk modeling and dashboarding, then attach ZIP-level SDOH features.
- Inputs:
  - OASISv2.csv created by preprocessing_v2 (Spark read).
  - ICD section and range mapping from preprocessing used to derive diabetes, heart failure, hypertension, and renal failure flags.
  - Texas SDOH ZIP, tract, and county reference file (texassdohtractgeorefcensus.csv).
- Processing:
  - Normalize and cast date fields; derive variables including AssessmentYear, NumVisits, PrevVisitDate, DaysBetweenVisits, LastAssessmentDate, and DaysCaredFor.
  - Map ICD‑10 codes to chronic condition flags using parsed ranges and a classifyconditions UDF across inpatient and primary or other diagnosis columns to create isdiabetes, isheartfailure, ishypertension, and isrenalfailure.
  - Filter to Texas records and select core features needed for modeling and dashboards, including demographics, clinical factors, utilization metrics, and ICD codes.
  - Compute BMI and BMICategory from Weightinpounds and Heightininches.
  - Join with the SDOH ZIP–county file to append county, population, urban or rural split, and ACS SDOH percentages such as education, broadband, income, poverty, and household structure.
- Outputs:
  - texasDashboardData.csv, a patient‑level Texas dataset that includes clinical variables, utilization metrics, BMI, chronic condition flags, and ZIP‑linked SDOH measures.

## 3. riskScoreSystem_v2.ipynb

- Purpose:
  - Build or apply a risk scoring system, such as a readmission risk score, using the dashboard‑ready clinical and SDOH features.
- Inputs:
  - texasDashboardData.csv or an equivalent dataset from DashboardPrep_v2 that contains READMISSION, chronic condition flags, BMI category, utilization metrics, and SDOH fields.
- Processing:
  - Engineer risk‑related features including chronic conditions, age, BMI, discipline, visit timing, care duration, and selected SDOH percentages.
  - Fit or apply a model or rules to compute patient‑level risk scores or risk tiers (for example, high, medium, and low risk for readmission).
- Outputs:
  - Patient‑level dataset with one or more risk score columns that can be consumed by ICD analysis and final dashboard data creation.

## 4. icdSectionAnalysis_v2.ipynb

- Purpose:
  - Analyze ICD sections and ranges, quantify condition burden across beneficiaries, and prepare ICD‑aggregated features for the dashboard and risk context.
- Inputs:
  - beneficiaryicds.csv and icdsections.csv generated in preprocessing_v2, each containing BeneficiaryID, ICDSection, ICDRange, and counts of sections by patient.
- Processing:
  - Use standardized ICD‑10 section ranges such as A00‑B99 through Z00‑Z99 to map each patient’s diagnoses to ICDSection and ICDRange values.
  - Aggregate these mappings to compute counts or indicators for each ICD range and section per beneficiary and visualize distributions such as the number of ICD sections per patient.
- Outputs:
  - ICDrangesperpatient.csv or an equivalent table with one row per BeneficiaryID and columns for each ICD range representing counts or presence.
  - ICD‑burden features that support risk interpretation and dashboard filtering by major disease category.

## 5. finalDashboardData_v2.ipynb

- Purpose:
  - Merge patient‑level clinical, utilization, SDOH, geographic, and ICD features to create the final dataset for the Connect‑2.0 dashboard.
- Inputs:
  - texasDashboardData.csv from DashboardPrep_v2.
  - US counties GeoJSON with FIPS codes and a state code lookup to identify Texas counties.
  - Aggregated county‑level metrics and ICD‑section features from earlier steps.
- Processing:
  - Load the US counties GeoJSON, convert it to a GeoDataFrame, and merge with the state code table to isolate Texas counties.
  - Join Texas county polygons to merged patient data using county name or county FIPS and then group by county and Fips to compute:
    - Unique counts such as BENEIDunique, FACINTIDunique, HHAAGNCYIDunique, and SBMHPSCDunique.
    - Chronic condition counts including Diabeticpatientcount, HeartFailurepatientcount, Hypertensionpatientcount, and RenalFailurepatientcount.
    - Readmission counts such as Readmission1count and Readmission0count.
    - County‑level SDOH and population metrics, including POPCOU, HOUCOU, POPURB, POPRUR, POPPCTURB, POPPCTRUR, and all ACS SDOH percentages.
    - Optional ICDSection and ICDRange summaries at the county level.
  - Export a streamlined county‑level CSV for interactive use in the dashboard application, such as streamlitdf.csv.
- Outputs:
  - Final geo‑enriched county‑level dashboard dataset with Texas county geometries, demographic and clinical metrics, chronic condition and readmission counts, SDOH measures, and ICD summaries ready for use in the Connect‑2.0 dashboard.
