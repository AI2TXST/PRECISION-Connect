import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Add current directory to path to allow importing fs_methods
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add parent directory to path if needed (though usually not for sibling imports)
sys.path.append(os.path.join(current_dir, '..'))

import fs_methods as fs
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

# Feature Description Mapping Helper
def get_feature_description(feature_name):
    common_desc = {
        'Age': 'Patient Age',
        'Gender': 'Patient Gender',
        'Race': 'Patient Race',
        'BMI': 'Body Mass Index',
        'has_diabetes': 'Indicator for Diabetes Diagnosis',
        'has_heart_failure': 'Indicator for Heart Failure Diagnosis',
        'has_hypertension': 'Indicator for Hypertension Diagnosis',
        'ever_readmitted': 'Patient was readmitted to facility',
    }
    if feature_name in common_desc:
        return common_desc[feature_name]
    if 'ICD' in feature_name:
        return "ICD Diagnostic Code Group"
    if 'score' in feature_name.lower():
        return "Risk Score or Calculated Metric"
    return "N/A"

# Tabs for Analysis and Comparison
tab1, tab2 = st.tabs(["Run Analysis", "Compare Results"])

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.header("Configuration")
    
    # 1. Cohort Selection
    cohort_options = ["Complete Cohort", "Diabetes", "Heart Failure", "Hypertension"]
    selected_cohort = st.selectbox("Select Cohort", cohort_options)

    # Filter Data based on Cohort (Logic stays here to update available targets)
    if selected_cohort == "Diabetes":
        df_fs = icdData[icdData['has_diabetes'] == 1].copy()
    elif selected_cohort == "Heart Failure":
        df_fs = icdData[icdData['has_heart_failure'] == 1].copy()
    elif selected_cohort == "Hypertension":
        df_fs = icdData[icdData['has_hypertension'] == 1].copy()
    else:
        df_fs = icdData.copy()
        
    st.write(f"Analyzing {len(df_fs)} patients.")

    # 2. Target Selection
    potential_targets = [c for c in df_fs.columns if 'score' in c.lower() or 'has_' in c.lower() or 'readmission' in c.lower() or 'ever_readmitted' in c]

    default_target_index = 0
    if 'ever_readmitted' in potential_targets:
        default_target_index = potential_targets.index('ever_readmitted')

    # Display Name Mapping
    target_display_map = {
        'ever_readmitted': 'Readmission',
        'readmission': 'Readmission',
        'has_diabetes': 'Diabetes',
        'has_heart_failure': 'Heart Failure',
        'has_hypertension': 'Hypertension',
    }

    def format_target_name(name):
        return target_display_map.get(name, name.replace('_', ' ').title())

    target_col = st.selectbox("Select Target Variable", sorted(potential_targets), 
                              index=default_target_index if 'ever_readmitted' in sorted(potential_targets) else 0,
                              format_func=format_target_name)

    # 3. Method Selection
    method_category = st.selectbox("Select Method Category", list(fs.METHOD_CATEGORIES.keys()))

    potential_methods = fs.METHOD_CATEGORIES[method_category]
    available_methods = [m for m in potential_methods if m in fs.AVAILABLE_METHODS]

    if available_methods:
        method_name = st.selectbox("Select Feature Selection Method", available_methods)
    else:
        st.warning(f"No methods available for {method_category}.")
        method_name = None
        
    run_clicked = st.button("Run Feature Selection")

# ----------------- MAIN CONTENT (TAB 1) -----------------
with tab1:
    # Guide / Explanations
    st.markdown("### Feature Selection Guide")
    with st.expander("Feature Selection Methods"):
        st.markdown("""
        **Filter Methods**
        * **Variance Threshold**: Removes all features whose variance doesn't meet some threshold. Low variance features carry listtle information.
        
        **Wrapper Methods**
        * **RFE (Recursive Feature Elimination)**: Fits a model and removes the weakest feature (or features) until the specified number of features is reached.
        * **SFS (Sequential Feature Selector)**: Adds (Forward) or removes (Backward) features to form a feature subset in a greedy fashion.
        
        **Embedded Methods**
        * **Lasso**: Linear model that penalizes the absolute magnitude of coefficients, often shrinking less important feature coefficients to zero.
        * **Random Forest / Gradient Boosting / XGBoost / LightGBM / CatBoost**: Ensemble learning methods that provide intrinsic measures of feature importance based on how much they improve the predictive model (e.g., Gini impurity decrease).
        
        **Permutation Methods**
        * **Permutation Importance**: Measures importance by calculating the increase in the model's prediction error after randomly shuffling the feature's values.
        """)

    with st.expander("Target Variables"):
        st.markdown("""
        * **Readmission**: Indicator of whether the patient was readmitted to the facility. Key metric for quality of care.
        * **Diabetes**: Patient diagnosed with diabetes.
        * **Heart Failure**: Patient diagnosed with heart failure.
        * **Hypertension**: Patient diagnosed with hypertension.
        """)
    
    st.divider()

    # Results Section
    if run_clicked and method_name:
        # Progress Bar Initialization
        fs_progress = st.progress(0, text="Initializing...")
        
        try:
            with st.spinner("Preprocessing and running..."):
                fs_progress.progress(10, text="Preprocessing Data...")
                # Preprocessing
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
                
                fs_progress.progress(30, text="Splitting Data...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                fs_func = fs.AVAILABLE_METHODS[method_name]
                
                fs_progress.progress(50, text=f"Running {method_name}...")
                results = fs_func(X_train, y_train, X_test, y_test)
                
                fs_progress.progress(80, text="Processing Results...")
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
                    
                # Download Button
                csv = res_df.to_csv(index=False).encode('utf-8')
                filename = f"fs_results_{method_name.replace(' ', '_')}_{target_col}.csv"
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=filename,
                    mime='text/csv',
                )
                
                fs_progress.progress(100, text="Analysis Complete!")
                
        except Exception as e:
            fs_progress.empty()
            st.error(f"Error running method: {e}")

with tab2:
    st.header("Compare Feature Selection Results")
    st.markdown("Upload two result CSV files generated by this tool to compare feature rankings and scores.")
    
    uploaded_files = st.file_uploader("Upload 2 Result CSVs", accept_multiple_files=True, type=['csv'])
    
    if len(uploaded_files) == 2:
        try:
            df1 = pd.read_csv(uploaded_files[0])
            df2 = pd.read_csv(uploaded_files[1])
            
            name1 = uploaded_files[0].name
            name2 = uploaded_files[1].name
            
            if 'Feature' in df1.columns and 'Feature' in df2.columns:
                # Merge on Feature Name
                if 'Score' in df1.columns:
                    df1 = df1.rename(columns={'Score': f'Score_{name1}'})
                if 'Score' in df2.columns:
                    df2 = df2.rename(columns={'Score': f'Score_{name2}'})
                    
                merged = pd.merge(df1[['Feature', f'Score_{name1}']], 
                                  df2[['Feature', f'Score_{name2}']], 
                                  on='Feature', how='outer').fillna(0)
                
                score_col1 = f"Score_{name1}"
                score_col2 = f"Score_{name2}"
                
                st.write("### Comparison Chart")
                
                col_c1, col_c2 = st.columns([2, 1])
                
                with col_c1:
                    # Scatter plot
                    fig_comp = px.scatter(merged, x=score_col1, y=score_col2, hover_data=['Feature'],
                                          title=f"Score Comparison: {name1} vs {name2}")
                    
                    # Add reference line
                    max_val = max(merged[score_col1].max(), merged[score_col2].max())
                    if max_val > 0:
                        fig_comp.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, 
                                          line=dict(color="Red", dash="dash"))
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                with col_c2:
                    st.write("### Top Common Features")
                    # Calculate average score or something?
                    merged['Avg_Score'] = (merged[score_col1] + merged[score_col2]) / 2
                    st.dataframe(merged.sort_values(by='Avg_Score', ascending=False).head(20))
                
            else:
                st.error("Uploaded files must contain a 'Feature' column.")
        except Exception as e:
            st.error(f"Error comparing files: {e}")
    elif len(uploaded_files) > 0 and len(uploaded_files) != 2:
        st.info("Please upload exactly 2 files to compare.")
