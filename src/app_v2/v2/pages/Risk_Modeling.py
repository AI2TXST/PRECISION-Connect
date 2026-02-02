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

# 1.5 Cohort Selection (Radio Button)
cohort_mode = st.sidebar.radio("Select Cohort", ["Complete Cohort", "Diabetic Patients", "Heart Failure Patients", "Hypertensive Patients"])

# Filter logic
if cohort_mode == "Complete Cohort":
    # No filtering
    pass
elif cohort_mode == "Diabetic Patients":
    icdData = icdData[icdData['has_diabetes'] == 1]
    st.sidebar.info(f"Filtered to {len(icdData)} Diabetic patients.")
elif cohort_mode == "Heart Failure Patients":
    icdData = icdData[icdData['has_heart_failure'] == 1]
    st.sidebar.info(f"Filtered to {len(icdData)} Heart Failure patients.")
elif cohort_mode == "Hypertensive Patients":
    icdData = icdData[icdData['has_hypertension'] == 1]
    st.sidebar.info(f"Filtered to {len(icdData)} Hypertensive patients.")

# Merge
with st.spinner("Merging Clinical and Risk Data..."):
    merged_df = pd.merge(icdData, risk_subset, on=['BENE_ID', 'Age'], how='inner')

st.write(f"**Data Status:** Merged {len(merged_df)} records using using **{selected_weight}** risk weights.")

if merged_df.empty:
    st.error("Merged dataset is empty. Check BENE_ID matching.")
    st.stop()

# --- Model Setup ---

# Target Variable Logic
potential_targets = ['ever_readmitted', 'has_diabetes', 'has_heart_failure', 'has_hypertension'] 
available_targets = [c for c in potential_targets if c in merged_df.columns]

# If no standard targets found, try fuzzy match
if not available_targets:
    available_targets = [c for c in merged_df.columns if 'readmit' in c.lower()]

# For Complete Cohort, allow predicting specific conditions (has_*)
# For specific cohorts, we restrict to readmission (handled below)
if cohort_mode == "Complete Cohort":
    has_targets = [c for c in merged_df.columns if c.startswith('has_')]
    # Add unique has_ targets
    available_targets.extend([t for t in has_targets if t not in available_targets])

# Sort for better UX
available_targets = sorted(list(set(available_targets)))

# Constraint: Limit target to Readmission for specific cohorts
if cohort_mode != "Complete Cohort":
    # Filter targets to only readmission related
    filtered_targets = [t for t in available_targets if 'readmit' in t.lower()]
    if filtered_targets:
        available_targets = filtered_targets
    else:
        st.warning("No readmission targets found for this cohort.")

selected_target = st.sidebar.selectbox("Select Target Predictor", available_targets)

# Feature Selection
# Drop ID columns and Target
ids_and_dates = [c for c in merged_df.columns if 'id' in c.lower() or 'date' in c.lower() or 'year' in c.lower()]

# Drop potential leakage columns
leakage_keywords = ['readmi', 'death', 'deceased', 'mortality', 'died', 'hosp']
leakage_cols = [c for c in merged_df.columns if any(k in c.lower() for k in leakage_keywords) and c != selected_target]

# Drop 'has_*' columns for specific cohorts to prevent leakage/redundancy as requested
has_cols = []
if cohort_mode != "Complete Cohort":
    has_cols = [c for c in merged_df.columns if 'has_' in c.lower()]

drop_cols = ['BENE_ID', 'Beneficiary_ID', 'ASMT_ID',  'weight', selected_target] + ids_and_dates + leakage_cols + has_cols

# Let user choose features or use all numeric
feature_mode = st.sidebar.radio("Feature Selection Mode", ["Auto (Numeric + Risk)", "Manual Selection"])

if feature_mode == "Manual Selection":
    all_features = [c for c in merged_df.columns if c not in drop_cols]
    # Features by category
    risk_feats = ['aids', 'alcohol', 'carit', 'chf', 'coag', 'cpd',
       'dane', 'depre', 'drug', 'fed', 'hypc', 'hypothy', 'hypunc', 'ld',
       'lymph', 'metacanc', 'obes', 'ond', 'para', 'pcd', 'psycho', 'pvd',
       'rf', 'rheumd', 'solidtum', 'valv', 'wloss', 'blane', 'diabc',
       'diabunc', 'pud', 'comorbidity_score', 'age_adj_comorbidity_score',
       'weight', 'ami', 'canc', 'cevd', 'copd', 'dementia', 'hp', 'mld',
       'rend', 'diab', 'diabwc', 'msld', 'hfrs', 'frail', 'impaired_hearing',
       'impaired_speech', 'impaired_vision', 'ld_asd']
    sdoh_feats = ['ACS_PCT_BACHELOR_DGR', 'ACS_PCT_COLLEGE_ASSOCIATE_DGR',
       'ACS_PCT_GRADUATE_DGR', 'ACS_PCT_HS_GRADUATE', 'ACS_PCT_LT_HS',
       'ACS_PCT_NO_WORK_NO_SCHL_16_19', 'ACS_PCT_POSTHS_ED',
       'ACS_PCT_VET_BACHELOR', 'ACS_PCT_VET_COLLEGE', 'ACS_PCT_VET_HS',
       'ACS_PCT_HH_LIMIT_ENGLISH', 'ACS_PCT_HH_BROADBAND',
       'ACS_PCT_HH_BROADBAND_ONLY', 'ACS_PCT_HH_CELLULAR',
       'ACS_PCT_HH_CELLULAR_ONLY', 'ACS_PCT_HH_DIAL_INTERNET_ONLY',
       'ACS_PCT_HH_INTERNET', 'ACS_PCT_HH_INTERNET_NO_SUBS',
       'ACS_PCT_HH_NO_COMP_DEV', 'ACS_PCT_HH_NO_INTERNET',
       'ACS_PCT_HH_OTHER_COMP', 'ACS_PCT_HH_OTHER_COMP_ONLY', 'ACS_PCT_HH_PC',
       'ACS_PCT_HH_PC_ONLY', 'ACS_PCT_HH_SAT_INTERNET',
       'ACS_PCT_HH_SMARTPHONE', 'ACS_PCT_HH_SMARTPHONE_ONLY',
       'ACS_PCT_HH_TABLET', 'ACS_PCT_HH_TABLET_ONLY',
       'ACS_PCT_CHILDREN_GRANDPARENT', 'ACS_PCT_CHILD_1FAM',
       'ACS_PCT_GRANDP_NO_RESPS', 'ACS_PCT_GRANDP_RESPS_NO_P',
       'ACS_PCT_GRANDP_RESPS_P', 'ACS_PCT_HH_1PERS', 'ACS_PCT_HH_ABOVE65',
       'ACS_PCT_HH_ALONE_ABOVE65', 'ACS_PCT_HH_KID_1PRNT',
       'ACS_TOT_GRANDCHILDREN_GP', 'ACS_PCT_HEALTH_INC_138_199',
       'ACS_PCT_HEALTH_INC_200_399', 'ACS_PCT_HEALTH_INC_ABOVE400',
       'ACS_PCT_HEALTH_INC_BELOW137', 'ACS_PCT_HH_1FAM_FOOD_STMP',
       'ACS_PCT_HH_FOOD_STMP_BLW_POV', 'ACS_PCT_HH_NO_FD_STMP_BLW_POV',
       'ACS_PCT_HH_PUB_ASSIST', 'ACS_PCT_INC50', 'ACS_PCT_INC50_ABOVE65',
       'ACS_PCT_NONVET_POV_18_64', 'ACS_PCT_PERSON_INC_100_124',
       'ACS_PCT_PERSON_INC_125_199', 'ACS_PCT_PERSON_INC_ABOVE200',
       'ACS_PCT_PERSON_INC_BELOW99', 'ACS_PCT_POV_AIAN', 'ACS_PCT_POV_ASIAN',
       'ACS_PCT_POV_BLACK', 'ACS_PCT_POV_HISPANIC', 'ACS_PCT_POV_MULTI',
       'ACS_PCT_POV_NHPI', 'ACS_PCT_POV_OTHER', 'ACS_PCT_POV_WHITE',
       'ACS_PCT_VET_POV_18_64', 'ACS_TOT_POP_POV']
    census_feats = ['COUNTY_NAME', 'POP_URB', 'POP_RUR',]
    clinical_feats = ['Submitted_HIPPS_Code',
       'Facility_Internal_ID', 'Age', 'Gender',
       'American_Indian_or_Alaska_Native', 'Asian',
       'Black_or_African_American', 'Hispanic_or_Latino',
       'Native_Hawiian_or_Pacific_Islander', 'White', 'ByDiscipline',
       'Days_Cared_For', 'BMI_Category']

            
    st.markdown("### Select Features by Category")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_risk = st.multiselect("Risk Scores", sorted(risk_feats), default=sorted(risk_feats)[:5] if risk_feats else [])
        sel_sdoh = st.multiselect("SDOH Factors", sorted(sdoh_feats))
    with col_f2:
        sel_census = st.multiselect("Demographics & Census", sorted(census_feats), default=[f for f in census_feats if 'age' in f.lower()])
        sel_clinical = st.multiselect("EHR / Clinical", sorted(clinical_feats))
        
    selected_features = sel_risk + sel_sdoh + sel_census + sel_clinical

else:
    # Auto: Select numeric types + some categoricals if low cardinality
    # For simplicity, let's take Risk Scores columns + some ICD aggregates
    # Risk Data cols: 'comorbidity_score', 'Age', etc.
    # We'll select numeric columns from the merged set
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = [c for c in numeric_cols if c not in drop_cols]
    st.info(f"Auto-selected {len(selected_features)} numeric features (Risk Scores + Comorbidities).")

if not selected_features:
    st.warning("No features selected.")
    st.stop()

# Prepare X and y
X = merged_df[selected_features].fillna(0) # Simple imputation
y = merged_df[selected_target]

# --- Training ---

import model_utils # Custom utils for KAN, Balancing, Logging

# Sidebar Hyperparameters & Model Selection
st.sidebar.subheader("Model Selection")

# Data Balancing Option
balance_data_opt = st.sidebar.checkbox("Balance Data (Undersampling)", value=False, 
                                   help="Balances the dataset by undersampling the majority class. Useful for rare events.")

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
    st.sidebar.markdown("### KAN Parameters")
    kan_epochs = st.sidebar.number_input("Epochs", 10, 500, 50)
    kan_batch = st.sidebar.selectbox("Batch Size", [128, 256, 512], index=1)
    kan_hidden = st.sidebar.number_input("Hidden Units", 10, 100, 20)
    kan_lr = st.sidebar.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.4f")

if st.button("Train Model", type="primary"):
    rm_progress = st.progress(0, text="Initializing Model...")
    
    # Balancing
    if balance_data_opt:
        rm_progress.progress(10, text="Balancing Data...")
        try:
            X_bal, y_bal = model_utils.balance_data(X, y)
            st.info(f"Balanced Data: {len(X_bal)} samples (Original: {len(X)})")
        except Exception as e:
            st.error(f"Balancing failed: {e}")
            X_bal, y_bal = X, y
    else:
        X_bal, y_bal = X, y

    with st.spinner(f"Training {model_type}..."):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
        rm_progress.progress(20, text="Splitting Data...")
        
        # Scale for KNN/NN/KAN
        if model_type in ["KNN (K-Nearest Neighbors)", "Kolmogorov-Arnold Network (KAN)"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # COnvert back to DF for SHAP if needed, but for KAN processing usually numpy/tensor is best.
            # SHAP expects DF for column names usually.
            X_train_np = X_train
            X_test_np = X_test
            X_train = pd.DataFrame(X_train, columns=X.columns) 
            X_test = pd.DataFrame(X_test, columns=X.columns)

        # Model Initialization & Training
        model = None # Standard sklearn-like models
        
        if model_type == "LightGBM":
            model = lgb.LGBMClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42, verbose=-1)
            model.fit(X_train, y_train)
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], depth=params['max_depth'], random_state=42, verbose=0)
            model.fit(X_train, y_train)
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
            model.fit(X_train, y_train)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42)
            model.fit(X_train, y_train)
        elif model_type == "KNN (K-Nearest Neighbors)":
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
            model.fit(X_train, y_train)
            
        elif model_type == "Kolmogorov-Arnold Network (KAN)":
            if not model_utils.TORCH_AVAILABLE:
                st.error("PyTorch not available. Cannot train KAN.")
                st.stop()
            
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            input_dim = X_train.shape[1]
            kan_model = model_utils.KAN(input_dim=input_dim, hidden_units=kan_hidden).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(kan_model.parameters(), lr=kan_lr, weight_decay=1e-4) # Fixed weight decay
            
            train_loader = DataLoader(model_utils.NumpyDataset(X_train_np, y_train.values), batch_size=kan_batch, shuffle=True)
            
            rm_progress.progress(30, text=f"Training KAN on {device}...")
            
            epoch_bar = st.empty()
            
            for epoch in range(kan_epochs):
                kan_model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = kan_model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Simple progress update
                if (epoch + 1) % 10 == 0:
                    epoch_bar.text(f"KAN Epoch {epoch+1}/{kan_epochs} - Loss: {train_loss:.4f}")
            
            rm_progress.progress(60, text="Evaluating KAN...")
            
            kan_model.eval()
            X_test_tensor = torch.as_tensor(X_test_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = kan_model(X_test_tensor).squeeze()
                y_prob = torch.sigmoid(logits).cpu().numpy()
                y_pred = (y_prob >= 0.5).astype(int)
            
            model = kan_model # Placeholder for shap (might need wrapper)

        elif model_type == "Stacked Ensemble":
            estimators = [
                ('lgbm', lgb.LGBMClassifier(n_estimators=params['n_estimators'], random_state=42, verbose=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=params['n_estimators'], use_label_encoder=False, eval_metric='logloss', random_state=42)),
                ('cb', cb.CatBoostClassifier(n_estimators=params['n_estimators'], verbose=0, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42))
            ]
            model = VotingClassifier(estimators=estimators, voting='soft')
            model.fit(X_train, y_train)

        rm_progress.progress(70, text="Evaluating Model...")
        
        # Standard Eval for non-KAN (KAN done above)
        if model_type != "Kolmogorov-Arnold Network (KAN)":
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        
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
            
        # --- Save Results ---
        if st.button("Save Results to CSV"):
            metrics_data = {
                "Model": model_type,
                "Cohort": cohort_mode,
                "Target": selected_target,
                "Accuracy": acc,
                "AUC": auc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
                "Timestamp": time.ctime()
            }
            try:
                saved_path = model_utils.save_model_results(metrics_data)
                st.success(f"Results saved to {saved_path}")
            except Exception as e:
                st.error(f"Failed to save results: {e}")
        
        rm_progress.progress(80, text="Calculating SHAP Explainability...")
        
        # --- Explainability (SHAP) ---
        st.subheader("🔍 Model Explainability (SHAP)")
        st.markdown("Understanding which risk factors contribute most to the prediction.")
        
        with st.spinner("Calculating SHAP values... (this may take a moment)"):
            try:
                # KAN SHAP Support
                if model_type == "Kolmogorov-Arnold Network (KAN)":
                    # For PyTorch models, use DeepExplainer or KernelExplainer
                    # Kernel is safer but slower. 
                    st.info("Using KernelExplainer for KAN (approximation)...")
                    # Use a small background sample
                    background = shap.kmeans(X_train_np, 10) if len(X_train_np) > 10 else X_train_np
                    
                    # Wrapper for prediction
                    def kan_predict(data):
                        t_data = torch.as_tensor(data, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            return torch.sigmoid(kan_model(t_data)).cpu().numpy()
                            
                    explainer = shap.KernelExplainer(kan_predict, background)
                    shap_values = explainer.shap_values(X_test_np[:50]) # Limit to 50 for speed
                    X_viz = pd.DataFrame(X_test_np[:50], columns=X.columns)
                    
                # Select Explainer for others
                elif model_type in ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Gradient Boosting"]:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    X_viz = X_test
                elif model_type in ["KNN (K-Nearest Neighbors)", "Stacked Ensemble"]:
                     # KernelExplainer is slow, use a subset of background data
                    background = shap.sample(X_train, 50) 
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_test.iloc[:50, :]) # explain only 50 samples for speed
                    X_test_viz = X_test.iloc[:50, :] # Use subset for viz
                    X_viz = X_test_viz
                else:
                    explainer = shap.Explainer(model)
                    shap_values = explainer.shap_values(X_test)
                    X_viz = X_test

                # Handle different SHAP return types (list for classification vs array)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Positive class
                elif hasattr(shap_values, "values"): # newer shap versions return Explanation object
                    if len(shap_values.shape) == 3: # (samples, features, classes)
                        shap_values = shap_values[:, :, 1]
                    else:
                        shap_values = shap_values.values
            
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
                    
            rm_progress.progress(100, text="Process Complete!")


