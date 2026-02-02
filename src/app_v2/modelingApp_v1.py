import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#import shap
import numpy as np
#import lime
#import lime.lime_tabular


st.set_page_config(
    page_title="Texas County Health System",
    page_icon='👩‍🏫',
    layout="wide",
    initial_sidebar_state="expanded")
#alt.themes.enable("dark")

#######################
# CSS styling - Modified to remove width constraints and reduce zoom
st.markdown("""
    <style>
    /* Remove sidebar width limit */
    .css-1d391kg {
        max-width: none;
    }

    /* Remove main content area width limit */
    .reportview-container {
        max-width: none;
        margin: 0;
    }
    
    /* Make the app use full viewport width */
    .main .block-container {
        max-width: none;
        padding: 1rem 2rem;
    }
    
    /* Reduce overall font sizes */
    .main .block-container {
        font-size: 14px;
    }
    
    /* Reduce header font sizes */
    h1 {
        font-size: 2rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
    }
    
    h4 {
        font-size: 1.1rem !important;
    }
    
    h5 {
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
        font-family: Helvetica Neue;
        font-size: 24px;
    }
    .header-container h1 {
        font-size: 24px;
        font-family: Helvetica Neue;
    }
    .header-container p {
        font-size: 16px;
        font-family: Helvetica Neue;
    }
    </style>
    <div class="header-container">
        <h1>Texan Readmission Analysis - Precision-Connect System</h1>
        <p>This interactive system allows users to select key features from a
        comprehensive dataset and apply feature selection methods. Users can
        also run machine learning models to analyze factors influencing hospital
        readmission rates for diabetic patients.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# **Sidebar**
with st.sidebar:
    st.title('👩‍🏫 Precision-Connect')
    with st.expander("About"):
        st.markdown("""
        **Key Features:**
        - Facility
        - Demographic
        - Geographic
        - Attaintment
        - English Fluency
        - Internet Connectivity
        - Living Conditions
        - Poverty
        - Comorbidity Scores

        **Created by:**
        - Mirna Elizondo (m_e172@txstate.edu)
        - Chloe Jones (bvd12@txstate.edu)
        - [Datalab12](https://datalab12.github.io/)
        - Texas State Center for Analytics and Data Science [TXST CADS](https://cads.txst.edu/)
        - [St. David's School of Nursing](https://www.nursing.txst.edu/)
        """)
    with st.expander("Data Sources"):
        st.markdown("""
        **Data Sources:**
        - The data used in this system is derived from publicly available sources, including state health departments and national demographic surveys.

        1. **CMS OASIS Data**
           - **Description**: Contains detailed information on home health patients, including demographics, diagnoses, care plans, and outcomes.
           - **Source**: [CMS OASIS Data](https://www.cms.gov/medicare/quality/home-health/oasis-data-sets)
           - **Use**: Analyze patient characteristics, treatment patterns, and home health outcomes.

        2. **Zip Code Data**
           - **Description**: Includes information on Texas zip codes, such as location, population, and area codes.
           - **Source**: [United States Zip Codes](https://www.unitedstateszipcodes.org/tx/)
           - **Use**: Map healthcare outcomes by region and segment other datasets geographically.

        3. **Census Urban Area Data**
           - **Description**: Provides data on urban areas, including population density and size.
           - **Source**: [Census 2020 Urban Area Data](https://www2.census.gov/geo/docs/reference/ua/2020_UA_COUNTY.xlsx)
           - **Use**: Segment populations into urban and rural areas to analyze healthcare access and outcomes.

        4. **Social Determinants of Health (SDOH) Data**
           - **Description**: Contains data on factors like income, education, and environment that influence health.
           - **Source**: [AHRQ SDOH Data](https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html)
           - **Use**: Integrate with other datasets to analyze the impact of social factors on health outcomes.
        """)


feature_categories = {
    "Demographic": ['Age', 'Gender', 'Race', 'ByDiscipline', 'DaysBetweenVisits',
                    'Days_Cared_For',  'BMI', 'BMI_Category'],
    "Geographic": [ 'COUNTY_NAME', 'POP_COU', 'HOU_COU',
                   'POP_URB', 'POPPCT_URB', 'POP_RUR', 'POPPCT_RUR'],
    "Attainment": ['ACS_PCT_BACHELOR_DGR', 'ACS_PCT_COLLEGE_ASSOCIATE_DGR',
                   'ACS_PCT_GRADUATE_DGR', 'ACS_PCT_HS_GRADUATE', 'ACS_PCT_LT_HS'],
    "Living Conditions": ['ACS_PCT_CHILDREN_GRANDPARENT', 'ACS_PCT_CHILD_1FAM',
                          'ACS_PCT_HH_1PERS', 'ACS_PCT_HH_ABOVE65'],
    "Poverty": ['ACS_PCT_HEALTH_INC_138_199', 'ACS_PCT_HEALTH_INC_200_399',
                'ACS_PCT_HEALTH_INC_ABOVE400', 'ACS_PCT_HEALTH_INC_BELOW137'],
    "Internet": ['ACS_PCT_HH_BROADBAND', 'ACS_PCT_HH_CELLULAR', 'ACS_PCT_HH_INTERNET'],
     "Comorbidity Scores": [
        "alcohol", "carit", "chf", "coag", "cpd", "dane", "depre", "drug", "fed",
        "hypc", "hypothy", "hypunc", "ld", "lymph", "metacanc", "obes", "ond", "para",
        "pcd", "psycho", "pvd", "rf", "rheumd", "solidtum", "valv", "wloss", "aids",
        "blane", "diabc", "diabunc", "pud", "comorbidity_score", "age_adj_comorbidity_score",
         "ami", "canc", "cevd", "copd", "dementia", "hp", "mld", "rend",
        "diab", "diabwc", "msld", "hfrs", "frail", "impaired_hearing",
        "impaired_speech", "impaired_vision", "ld_asd"
    ]
    #"Facility": ['SBMHPSCD', 'FACINTID', 'HHA_AGNCY_ID', 'M0010'],
    #"Codes": ['M1021_PRI_DGN_ICD', 'M1023_OTH_DGN1_ICD', 'M1023_OTH_DGN2_ICD',
    #          'M1023_OTH_DGN3_ICD', 'M1023_OTH_DGN4_ICD', 'M1023_OTH_DGN5_ICD_I10'],
    #"ICD Info": ['ICD_Section', 'ICD_Range'],
}

condition_labels = ['HeartFailure_patient', 'Hypertension_patient', 'RenalFailure_patient']
target = ['Readmission']
ignore_columns = condition_labels + target
#uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Load Data
current_path = os.getcwd()
csv_path = os.path.join(current_path, 'data/diabeticDashboardData_ICDSections.csv')
risk_path = os.path.join(current_path, 'data/diabeticsRiskScores.csv')

# Check if the file exists
if os.path.exists(csv_path):
    df_merge = pd.read_csv(csv_path, low_memory=False).drop(['ASMTFFDT','Diabetic_patient', 'primary_city', 'area_codes', 'COUNTYFIPS', "M1060_HEIGHT", "M1060_WEIGHT"], axis=1)
    df_risk = pd.read_csv(risk_path, low_memory=False)
    df_risk = df_risk[df_risk['weight'] == 'Swiss']
    df_risk = df_risk.drop('weight', axis=1)
    df = pd.merge(df_merge, df_risk, on=['BENE_ID', 'Age'], how='left')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Data Preview")
        st.dataframe(df.head(20))
    with col2:
        st.write('### Readmission Counts:')
        fig = px.histogram(df, x="Readmission", color="Readmission", title="Readmission Counts", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


else:
    st.error("Missing required data files. Please check the file paths.")
    st.stop()

if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "model_run" not in st.session_state:
    st.session_state.model_run = False

# Header
st.header("Select Feature Groups")

# Let user select the feature groups they want to include
selected_groups = st.multiselect(
    "Select the Feature Groups to Include",
    options=list(feature_categories.keys()),
    default=['Demographic']
)

if selected_groups != st.session_state.selected_features:
    st.session_state.selected_features = []
    for group in selected_groups:
        st.session_state.selected_features.extend(feature_categories[group])

selected_features = st.session_state.selected_features

# Display the selected features (you can also show this in a more compact way, as discussed earlier)
if selected_features:
    st.write("### Your Selected Features:")
    st.write(list(selected_features))  # Or use a list view for better readability
else:
    st.write("No features selected.")

# Continue button (to move to the next step)
# Display selected features
if selected_features:
    st.write("##### Class balancing and scaling features")
    st.session_state.model_run = False

    X = df[selected_features]
    y = df['Readmission']
    #first_category_key = feature_categories['Facility']
    #X[first_category_key] = X[first_category_key].astype(str)
    # Identify categorical and numerical columns
    num_columns = X.select_dtypes(include=['number']).columns
    cat_columns = X.select_dtypes(include=['object', 'category']).columns
    X[num_columns] = X[num_columns].apply(lambda col: col.fillna(col.mean()), axis=0)

    for column in cat_columns:
        X[column] = X[column].fillna("")


    st.write("##### Feature Type Summary")
    col1, col2, col3 = st.columns([0.2, 0.2, 1])
    with col1:
        st.write(f"**Numerical Columns ({len(num_columns)}):**", num_columns.tolist())
    with col2:
        st.write(f"**Categorical Columns ({len(cat_columns)}):**", cat_columns.tolist())
    with col3:
        selected_df = X[num_columns]
        summary_stats = selected_df.describe()
        st.write("##### Summary Statistics for Selected Numeric Features")
        st.dataframe(summary_stats)

    if not cat_columns.empty:
        one = OneHotEncoder(sparse=False, drop='first')
        X_cat_encoded = one.fit_transform(X[cat_columns])

        # Manually generate meaningful column names for the encoded categorical features
        encoded_feature_names = [
            f"{col}_{val}" for col, values in zip(cat_columns, one.categories_) for val in values[1:]
        ]

        # Convert the one-hot encoded data into a DataFrame with meaningful column names
        X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=encoded_feature_names, index=X.index)
        X = pd.concat([X_cat_encoded_df, X[num_columns]],axis=1 )
    else:
        X = X[num_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    over = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X_train_resampled, y_train_resampled = over.fit_resample(X, y)

    # Scale the features (important: fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    st.write(f"#### Training data shape: {X_train_scaled.shape}, Test data shape: {X_test_scaled.shape}")

    st.success("Data processing complete! Now, run the LightGBM model.")
    # Model button
    if st.button("Run LightGBM Model"):
        if not st.session_state.model_run:
            st.session_state.model_run = True
        # LightGBM model
            model = lgb.LGBMClassifier()
            param_dist = {
            'num_leaves': [31, 50, 70, 100],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 500],
            'max_depth': [-1, 5, 10, 15],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 50]
            }

            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                               n_iter=50, cv=3, verbose=1, random_state=42, n_jobs=-1)
            progress_bar = st.progress(0)
            st.write("### Tuning Hyperparameters with RandomizedSearchCV...")
            with st.spinner("Performing hyperparameter tuning..."):
                random_search.fit(X_train_scaled, y_train_resampled)
                for i in range(10):
                    time.sleep(0.5)
                    progress_bar.progress(int((i + 1) * 10))

            st.write("### Hyperparameter Tuning Complete!")
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            st.write("Best Hyperparameters:", best_params)
            feature_importance = best_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            })
            importance_df = importance_df[importance_df['Importance'] > 50]
            importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
            model = best_model.fit(X_train_scaled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Display the metrics
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }

            # Create a DataFrame for plotting
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
            col1, col2, col3 = st.columns([2, 3, 3])  # Adjust ratio if needed

            # Feature Importance Plot (LightGBM)
            with col1:
                st.write("### Feature Importance")
                fig1, ax1 = plt.subplots(figsize=(4, len(importance_df)/2))  # Adjust height dynamically
                ax1.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
                ax1.set_xlabel("Importance Score")
                ax1.set_title("Feature Importance (LightGBM)")
                st.pyplot(fig1)
            with col2:
                st.write("### Model Performance")
                fig3 = go.Figure()
                for i, (metric, score) in enumerate(metrics.items()):
                    fig3.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        title={'text': metric},
                        gauge={'axis': {'range': [0, 1]}},
                        domain={'row': i // 2, 'column': i % 2}
                    ))

                fig3.update_layout(
                    grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
                    title_text="Gauge Metrics",
                    width=500, height=500
                )
                st.plotly_chart(fig3)
            with col3:
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write("### Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Readmission', 'Readmission'], yticklabels=['No Readmission', 'Readmission'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                st.pyplot(fig)
        else:
            st.warning("LightGBM model has already been run. Refresh to restart.")
    else:
        st.warning("Please run model.")
else:
    st.warning("Please select features.")
'''
            st.subheader("Feature Importance (SHAP)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.yticks(rotation=90)
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig, bbox_inches='tight')
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:5]  # top 5
            top_features = X_test.columns[top_indices]
            for feature in top_features:
                st.write(f"Dependence plot for: {feature}")
                fig = plt.figure(figsize=(6, 3))
                shap.dependence_plot(feature, shap_values, X_test, show=False)
                st.pyplot(fig)

            # 5. Plot heatmap of SHAP values (optional overview)
            st.write("SHAP values heatmap")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap_values_matrix = shap_values if isinstance(shap_values, np.ndarray) else shap_values.values
            cax = ax.imshow(shap_values_matrix, aspect='auto', interpolation='nearest', cmap='coolwarm')
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Sample Index")
            fig.colorbar(cax)
            st.pyplot(fig)
'''
