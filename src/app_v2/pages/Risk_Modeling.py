import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import matplotlib.pyplot as plt

# Try imports for ML
try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    import shap
    ML_AVAILABLE = True
except ImportError as e:
    ML_ERROR = str(e)
    ML_AVAILABLE = False


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier # For KAN proxy


st.set_page_config(page_title="Risk Modeling", page_icon="🤖", layout="wide")

st.title("🤖 Patient Risk Modeling & Explainability")
st.markdown("""
This module integrates patient clinical data with **risk scores** (Charlson, Elixhauser/Van Walraven, etc.) 
to predict adverse outcomes using **LightGBM**. It uses **SHAP (SHapley Additive exPlanations)** to interpret model decisions.
""")

if not ML_AVAILABLE:
    st.error(f"⚠️ **ML Libraries Missing**: `lightgbm`, `xgboost`, `catboost` or `shap` could not be imported.")
    if 'ML_ERROR' in locals():
        st.code(f"Error Details: {ML_ERROR}")
    st.info(f"**Current Python Environment:** `{sys.executable}`")
    st.stop()
    


# --- Data Loading ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ../v2/
    # Adjust paths based on known structure
    # v2/pages -> v2 -> dashboardData...
    # files seem to be in `oasis_data/version2/` or similar relative to v2
    
    # Try finding the root
    # If file is c:/.../v2/pages/Risk_Modeling.py
    # Root is c:/.../v2/
    
    # Paths from previous exploration
    icd_relt_path = "../../dashboardData_ICDSections_v2.csv"
    risk_relt_path = "../../patientRiskScores.csv"
    
    if not os.path.exists(os.path.join(base_path, icd_relt_path)):
       # Fallback if running from a different CWD
       icd_relt_path = "../dashboardData_ICDSections_v2.csv" 
       risk_relt_path = "../patientRiskScores.csv"

    # Construct absolute paths to be safe if possible, or trust relative
    # Let's try loading
    try:
        icd_df = pd.read_csv(icd_relt_path, low_memory=False)
        risk_df = pd.read_csv(risk_relt_path, low_memory=False)
        return icd_df, risk_df
    except FileNotFoundError:
        st.error(f"Data files not found. Checked: {icd_relt_path} and {risk_relt_path}")
        return None, None

icdData, riskData = load_data()

if icdData is None or riskData is None:
    st.stop()

# --- Preprocessing & Merging ---

st.sidebar.header("Configuration")

# 1. Risk Weight System Selection
risk_weights = riskData['weight'].unique()
selected_weight = st.sidebar.selectbox("Select Risk Weight System", risk_weights, index=0)

# Filter Risk Data
risk_subset = riskData[riskData['weight'] == selected_weight].copy()

# 2. Merge Data
# icdData has 'Beneficiary_ID', riskData has 'BENE_ID'
# Standardize keys
if 'Beneficiary_ID' in icdData.columns:
    icdData['BENE_ID'] = icdData['Beneficiary_ID']

# 1.5 Cohort Selection (Filter)
cohort_options = ["Diabetes", "Heart Failure", "Hypertension"]
selected_cohorts = st.sidebar.multiselect("Filter Cohort (Include Patients with...)", cohort_options, default=[])

if selected_cohorts:
    # Map friendly names to column names
    cohort_map = {
        "Diabetes": "has_diabetes",
        "Heart Failure": "has_heart_failure",
        "Hypertension": "has_hypertension"
    }
    
    # Filter logic: Union (OR) or Intersection (AND)? 
    # Usually "Show me patients with Diabetes OR Hypertension".
    # But if Multiselect, 'AND' effectively narrows it down to specific co-morbidities. 
    # Let's assume OR for broad inclusion, but let's check user intent "combo".
    # A safe bet is "Patients having ANY of the selected".
    # Implementation: mask = col1 | col2 | ...
    
    mask = pd.Series(False, index=icdData.index)
    col_found = False
    for cohort in selected_cohorts:
        col = cohort_map.get(cohort)
        if col and col in icdData.columns:
            mask = mask | (icdData[col] == 1)
            col_found = True
            
    if col_found:
        icdData = icdData[mask]
        st.sidebar.info(f"Filtered to {len(icdData)} patients in selected cohorts.")
    else:
        st.sidebar.warning("Selected cohort columns not found in data.")

# Merge
with st.spinner("Merging Clinical and Risk Data..."):
    merged_df = pd.merge(icdData, risk_subset, on='BENE_ID', how='inner')

st.write(f"**Data Status:** Merged {len(merged_df)} records using using **{selected_weight}** risk weights.")

if merged_df.empty:
    st.error("Merged dataset is empty. Check BENE_ID matching.")
    st.stop()

# --- Model Setup ---

# Target Variable
potential_targets = ['ever_readmitted', 'ever_deceased', 'READMISSION'] # Modify based on actual cols
available_targets = [c for c in potential_targets if c in merged_df.columns]
if not available_targets:
    available_targets = [c for c in merged_df.columns if 'readmit' in c.lower() or 'death' in c.lower() or 'has_' in c.lower()]

selected_target = st.sidebar.selectbox("Select Target Predictor", available_targets)

# Feature Selection
# Drop ID columns and Target
ids_and_dates = [c for c in merged_df.columns if 'id' in c.lower() or 'date' in c.lower() or 'year' in c.lower()]
# Exclude known good features that might contain 'id' or 'date' if any (e.g. 'valid_date_flag'?) - usually none in this dataset context.

# Drop potential leakage columns (variations of readmission/death if not the target)
leakage_keywords = ['readmi', 'death', 'deceased', 'mortality', 'died', 'hosp']
leakage_cols = [c for c in merged_df.columns if any(k in c.lower() for k in leakage_keywords) and c != selected_target]

drop_cols = ['BENE_ID', 'Beneficiary_ID', 'ASMT_ID', 'CASE_ID', 'weight', selected_target] + ids_and_dates + leakage_cols

# Also explicitly drop raw diag codes if they are high cardinality and not useful for LightGBM numeric input (unless encoded)
# Assuming we mostly want numeric risk scores + flags


# Let user choose features or use all numeric
feature_mode = st.sidebar.radio("Feature Selection Mode", ["Auto (Numeric + Risk)", "Manual Selection"])

if feature_mode == "Manual Selection":
    all_features = [c for c in merged_df.columns if c not in drop_cols]
    selected_features = st.multiselect("Select Input Features", all_features, default=all_features[:10])
else:
    # Auto: Select numeric types + some categoricals if low cardinality
    # For simplicity, let's take Risk Scores columns + some ICD aggregates
    # Risk Data cols: 'comorbidity_score', 'Age', etc.
    # We'll select numeric columns from the merged set
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = [c for c in numeric_cols if c not in drop_cols]

if not selected_features:
    st.warning("No features selected.")
    st.stop()

# Prepare X and y
X = merged_df[selected_features].fillna(0) # Simple imputation
y = merged_df[selected_target]

# --- Training ---

# Sidebar Hyperparameters & Model Selection
st.sidebar.subheader("Model Selection")
model_type = st.sidebar.selectbox("Select Model", 
    ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Gradient Boosting", 
     "KNN (K-Nearest Neighbors)", "Stacked Ensemble", "Kolmogorov-Arnold Network (KAN)"])

# Hyperparameters
params = {}
if model_type in ["LightGBM", "XGBoost", "CatBoost", "Gradient Boosting", "Random Forest", "Stacked Ensemble"]:
    params['n_estimators'] = st.sidebar.slider("N Estimators", 50, 1000, 100)
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1) if model_type != "Random Forest" else None

if model_type == "KNN":
    params['n_neighbors'] = st.sidebar.slider("K Neighbors", 3, 50, 5)

if model_type == "Kolmogorov-Arnold Network (KAN)":
    st.sidebar.info("Note: Using an MLP approximation for KAN in this environment.")
    params['hidden_layer_sizes'] = (100, 50)
    params['learning_rate_init'] = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)

if st.button("Train Model", type="primary"):
    with st.spinner(f"Training {model_type}..."):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale for KNN/NN
        if model_type in ["KNN (K-Nearest Neighbors)", "Kolmogorov-Arnold Network (KAN)"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=X.columns) # Keep cols for SHAP
            X_test = pd.DataFrame(X_test, columns=X.columns)

        # Model Initialization
        if model_type == "LightGBM":
            model = lgb.LGBMClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42, verbose=-1)
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], depth=params['max_depth'], random_state=42, verbose=0)
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42)
        elif model_type == "KNN (K-Nearest Neighbors)":
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif model_type == "Kolmogorov-Arnold Network (KAN)":
            # Using MLP as proxy
            model = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'], random_state=42, max_iter=500)
        elif model_type == "Stacked Ensemble":
            estimators = [
                ('lgbm', lgb.LGBMClassifier(n_estimators=params['n_estimators'], random_state=42, verbose=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=params['n_estimators'], use_label_encoder=False, eval_metric='logloss', random_state=42)),
                ('cb', cb.CatBoostClassifier(n_estimators=params['n_estimators'], verbose=0, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42))
            ]
            # Majority voting usually implies hard voting, but soft is better for AUC/Probabilities
            # User asked for Majority Voting (Hard), but we also want AUC. 
            # 'soft' voting averages probabilities, 'hard' votes inputs.
            # Let's use 'soft' to keep the rest of the flow (AUC/SHAP) working smoothly, 
            # effectively a weighted average ensemble.
            model = VotingClassifier(estimators=estimators, voting='soft')

        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Display Metrics
        st.subheader("Model Performance")
        
        # Row 1: Key Stats
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("Accuracy", f"{acc:.2%}")
        with m2: st.metric("ROC AUC", f"{auc:.3f}")
        with m3: st.metric("Precision", f"{prec:.3f}")
        with m4: st.metric("Recall", f"{rec:.3f}")
        with m5: st.metric("F1 Score", f"{f1:.3f}")
        
        st.write("---")
        
        # Row 2: Confusion Matrix Details & Visual
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("#### Confusion Matrix Stats")
            st.write(f"**True Positives (TP):** {tp}")
            st.write(f"**True Negatives (TN):** {tn}")
            st.write(f"**False Positives (FP):** {fp}")
            st.write(f"**False Negatives (FN):** {fn}")
            
        with c2:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix Visualization")
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # --- Explainability (SHAP) ---
        st.subheader("🔍 Model Explainability (SHAP)")
        st.markdown("Understanding which risk factors contribute most to the prediction.")
        
        with st.spinner("Calculating SHAP values... (this may take a moment)"):
            try:
                # Select Explainer
                if model_type in ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Gradient Boosting"]:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                elif model_type in ["KNN (K-Nearest Neighbors)", "Kolmogorov-Arnold Network (KAN)", "Stacked Ensemble"]:
                     # KernelExplainer is slow, use a subset of background data
                    background = shap.sample(X_train, 50) 
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_test.iloc[:50, :]) # explain only 50 samples for speed
                    X_test_viz = X_test.iloc[:50, :] # Use subset for viz
                else:
                    explainer = shap.Explainer(model)
                    shap_values = explainer.shap_values(X_test)

                # Handle different SHAP return types (list for classification vs array)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Positive class
                elif hasattr(shap_values, "values"): # newer shap versions return Explanation object
                    if len(shap_values.shape) == 3: # (samples, features, classes)
                        shap_values = shap_values[:, :, 1]
                    else:
                        shap_values = shap_values.values

                # Correct X for visualization if we subsampled
                X_viz = X_test_viz if 'X_test_viz' in locals() else X_test

            except Exception as e:
                st.warning(f"SHAP explanation failed or is not supported for this model: {e}")
                shap_values = None

            if shap_values is not None:
                # Layout: Side-by-Side Plots
                col_shap1, col_shap2 = st.columns(2)
                
                with col_shap1:
                    st.markdown("#### Feature Importance Summary")
                    fig_shap, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_viz, plot_type="dot", show=False)
                    st.pyplot(fig_shap)
                    
                with col_shap2:
                    st.markdown("#### Feature Dependence Plot")
                    # Auto-select feature
                    if isinstance(shap_values, np.ndarray):
                        vals = np.abs(shap_values).mean(0)
                    else:
                        vals = np.abs(shap_values.values).mean(0)
                        
                    feature_importance = pd.DataFrame(list(zip(X_viz.columns, vals)), columns=['col_name','feature_importance_vals'])
                    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
                    top_features = feature_importance['col_name'].tolist()
                    
                    # Dropdown for specific feature
                    dep_feature = st.selectbox("Select Feature", top_features, index=0)
                    
                    fig_dep, ax_dep = plt.subplots(figsize=(10, 8))
                    shap.dependence_plot(dep_feature, shap_values, X_viz, ax=ax_dep, show=False)
                    st.pyplot(fig_dep)


