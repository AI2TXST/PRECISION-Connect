import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Add parent directory to path to allow importing featureSelection
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import featureSelection as fs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Feature Selection", page_icon="🔍", layout="wide")

# Load Data
# Replicating data loading from app.py
# Assuming CWD is the project root (SYSTEM_v2) if run via `streamlit run v2/app.py`
# or handling the relative paths carefully.

@st.cache_data
def load_data():
    # Try different potential paths
    possible_roots = [
        "../../", # if in v2/pages and running from v2/pages (unlikely)
        "../",    # if in v2/pages and running from v2
        "./",     # if running from root and files are here
    ]
    
    # Based on app.py:
    # csv_path = "../../streamlit_df.csv"
    # icd_path = "../../dashboardData_ICDSections_v2.csv"
    
    # If running "streamlit run v2/app.py" from SYSTEM_v2:
    # app.py found files at "../../"? That means they are in `oasis_data/version2`?
    # Let's stick to what worked in app.py:
    
    icd_path = "../../dashboardData_ICDSections_v2.csv"
    if not os.path.exists(icd_path):
        icd_path = "../dashboardData_ICDSections_v2.csv"
    
    if os.path.exists(icd_path):
        icdData = pd.read_csv(icd_path, low_memory=False)
        return icdData
    else:
        st.error(f"Data file not found at {icd_path}. PWD: {os.getcwd()}")
        return pd.DataFrame()

icdData = load_data()

st.title("Feature Selection Analysis")
st.markdown("Select a cohort, a target variable, and a method to identify predictors.")

if icdData.empty:
    st.stop()

# 1. Cohort Selection
cohort_options = ["Complete Cohort", "Diabetes", "Heart Failure", "Hypertension"]
selected_cohort = st.selectbox("Select Cohort", cohort_options)

# Filter Data based on Cohort
if selected_cohort == "Diabetes":
    df_fs = icdData[icdData['has_diabetes'] == 1].copy()
elif selected_cohort == "Heart Failure":
    df_fs = icdData[icdData['has_heart_failure'] == 1].copy()
elif selected_cohort == "Hypertension":
    df_fs = icdData[icdData['has_hypertension'] == 1].copy()
else:
    df_fs = icdData.copy()
    
st.write(f"Analyzing {len(df_fs)} patients in **{selected_cohort}**.")

# 2. Target Selection
potential_targets = [c for c in df_fs.columns if 'score' in c.lower() or 'has_' in c.lower() or 'readmission' in c.lower() or 'ever_readmitted' in c]

default_target_index = 0
if 'ever_readmitted' in potential_targets:
    default_target_index = potential_targets.index('ever_readmitted')
elif 'ever_readmitted' in df_fs.columns:
    # If it wasn't picked up by logic but exists? (It should be picked up by 'readmission')
    pass

target_col = st.selectbox("Select Target Variable", sorted(potential_targets), index=default_target_index if 'ever_readmitted' in sorted(potential_targets) else 0)

# 3. Method Selection
method_category = st.selectbox("Select Method Category", list(fs.METHOD_CATEGORIES.keys()))

potential_methods = fs.METHOD_CATEGORIES[method_category]
available_methods = [m for m in potential_methods if m in fs.AVAILABLE_METHODS]

if available_methods:
    method_name = st.selectbox("Select Feature Selection Method", available_methods)
else:
    st.warning(f"No methods available for {method_category}.")
    method_name = None

# Feature Description Mapping Helper (Copied from app.py)
# Ideally could be in a shared utils file.
def get_feature_description(feature_name):
    # Shortened version or duplicated
    common_desc = {
        'Age': 'Patient Age',
        'Gender': 'Patient Gender',
        'Race': 'Patient Race',
        'BMI': 'Body Mass Index',
        'has_diabetes': 'Indicator for Diabetes Diagnosis',
        'has_heart_failure': 'Indicator for Heart Failure Diagnosis',
        'has_hypertension': 'Indicator for Hypertension Diagnosis',
        'readmission': 'Readmission Indicator',
        'ever_readmitted': 'Patient was readmitted to facility',
        'death': 'Mortality Indicator',
    }
    if feature_name in common_desc:
        return common_desc[feature_name]
    if 'ICD' in feature_name:
        return "ICD Diagnostic Code Group"
    if 'score' in feature_name.lower():
        return "Risk Score or Calculated Metric"
    return "N/A"

if st.button("Run Feature Selection") and method_name:
    with st.spinner("Preprocessing and running..."):
        # Preprocessing
        # Drop IDs etc
        cols_to_drop = [c for c in df_fs.columns if 'ID' in c or 'Name' in c or 'Date' in c or 'Code' in c]
        cols_to_drop.append('County') if 'County' in df_fs.columns else None
        cols_to_drop.append('COUNTY_NAME') if 'COUNTY_NAME' in df_fs.columns else None
        
        if target_col in cols_to_drop:
            cols_to_drop.remove(target_col)
        
        X = df_fs.drop(columns=cols_to_drop, errors='ignore').drop(columns=[target_col], errors='ignore')
        y = df_fs[target_col]

        # Encoding
        le = LabelEncoder()
        for col in X.select_dtypes(include='object').columns:
            X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.fillna(0)
        y = y.fillna(0)
        
        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        fs_func = fs.AVAILABLE_METHODS[method_name]
        
        try:
            results = fs_func(X_train, y_train, X_test, y_test)
            st.success(f"Completed {method_name}")
            
            feature_names = X.columns
            selected_indices = results['indices']
            scores = results['scores']
            
            res_data = []
            for idx, score in zip(selected_indices, scores):
                res_data.append({
                    "Feature": feature_names[idx],
                    "Score": score,
                    "Description": get_feature_description(feature_names[idx])
                })
            
            res_df = pd.DataFrame(res_data).sort_values(by="Score", ascending=False)
            
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                st.markdown("### Top 20 Features")
                top_df = res_df.head(20)
                fig_bar = px.bar(top_df, x='Score', y='Feature', orientation='h', 
                                 title=f"Feature Importance ({results['score_type']})",
                                 hover_data=['Description'])
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_res2:
                st.markdown("### Selection Details")
                st.dataframe(res_df)
                
            # Option to save results?
            if st.button("Save Results"):
                import os
                output_dir = "feature_selection_results"
                os.makedirs(output_dir, exist_ok=True)
                fs.save_feature_selection_results(results, feature_names, output_dir=output_dir)
                st.success(f"Results saved to {output_dir}")
            
        except Exception as e:
            st.error(f"Error running method: {e}")
