import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import json
import altair as alt
import os
from PIL import Image
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff
import numpy as np

CHART_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
               '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

st.set_page_config(
    page_title="County Comorbidity Analysis",
    page_icon='👩‍🏫',
    layout="wide",
    initial_sidebar_state="expanded",
)
#alt.themes.enable("dark")

#######################
# CSS styling
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
    
    /* Print Styles */
    @media print {
        /* Hide sidebar, header decorations, and footer */
        section[data-testid="stSidebar"] { display: none; }
        header { display: none; }
        footer { display: none; }
        .stDeployButton { display: none; }
        
        /* Maximize width */
        .main .block-container {
            max-width: 100%;
            padding: 0;
            margin: 0;
        }
        
        /* Adjust font sizes for print */
        html, body, [class*="css"] {
            font-size: 12px;
        }
        
        h1, h2, h3 { 
            margin-top: 0; 
            margin-bottom: 0.5rem; 
        }
        
        /* Force charts to fit */
        .js-plotly-plot {
            page-break-inside: avoid;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Load Data
# Load Data
current_path = os.getcwd()
# Assuming the app is run from the parent directory or consistent relative paths
csv_path = "../../streamlit_df.csv"
geo_path = "../data/texas_counties.geojson" # This seems to be in v2/data based on my list_dir
disparity_path = "../../streamlit_sdoh.csv"
icd_path = "../../dashboardData_ICDSections_v2.csv"

# Check if the file exists
if os.path.exists(csv_path) and os.path.exists(disparity_path) and os.path.exists(icd_path):
    grouped_df = pd.read_csv(csv_path)
    disparity_df = pd.read_csv(disparity_path)
    icdData = pd.read_csv(icd_path, low_memory=False)
    with open(geo_path, "r") as f:
        texas_counties_geojson = json.load(f)
else:
    # Fallback to local paths if running from v2 folder directly without parent context
    # Try absolute paths or check where we are
    st.write(f"Debug: CWD is {os.getcwd()}")
    st.error("Missing required data files. Please check the file paths.")
    st.stop()


def extract_category_counts(df, column_name):
    """ Extracts category counts from a column of dictionaries stored as strings and flattens them into a DataFrame. """

    # Debug: check what the column contains
    #st.write(f"Contents of column {column_name}:")
    #st.write(df[column_name].head())

    category_counts = []

    # Iterate through the column of string representations of dictionaries
    for cat_dict_str in df[column_name]:
        try:
            # Replace single quotes with double quotes for valid JSON format
            if isinstance(cat_dict_str, str):
                cat_dict_str = cat_dict_str.replace("'", '"')  # Replace single quotes with double quotes
                cat_dict = json.loads(cat_dict_str)
            else:
                cat_dict = cat_dict_str

            # For each dictionary, create a list of (category, count) pairs
            if isinstance(cat_dict, dict):
                for category, count in cat_dict.items():
                    category_counts.append({"Category": category, "Count": count})
            else:
                # If it's not a dictionary, append an empty dictionary (or handle differently if needed)
                category_counts.append({"Category": None, "Count": 0})
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error decoding dictionary: {e}")
            category_counts.append({"Category": None, "Count": 0})

    # Convert to DataFrame
    counts_df = pd.DataFrame(category_counts)

    # Debug: check the resulting DataFrame
    #st.write("Extracted category counts:")
    #st.write(counts_df.head())

    return counts_df


def aggregate_patient_data(df):
    """ Aggregates patient-level data to match county-level grouped_df structure. """
    if df.empty:
        return pd.DataFrame()

    # Define aggregation logic
    agg_funcs = {
        'Beneficiary_ID': 'nunique', # BENE_ID_unique
        'Facility_Internal_ID': 'nunique', # FACINTID_unique
        'Submitted_HIPPS_Code': 'nunique', # SBMHPSCD_unique
        'Age': 'mean', # Age_mean
        'Agency_Medicare_Number': 'nunique', # M0010_counts (count of agencies)
    }

    # Group by County (using COUNTY_NAME)
    # We need to preserve POP_URB, POPPCT_URB, POP_RUR, POPPCT_RUR as they are constant for county
    # We'll take the first value for these
    spatial_cols = ['POP_URB', 'POPPCT_URB', 'POP_RUR', 'POPPCT_RUR', 'COUNTYFIPS']
    
    # Process numeric aggregations
    grouped = df.groupby('COUNTY_NAME').agg(agg_funcs).reset_index()
    grouped = grouped.rename(columns={
        'COUNTY_NAME': 'County',
        'Beneficiary_ID': 'BENE_ID_unique',
        'Facility_Internal_ID': 'FACINTID_unique',
        'Submitted_HIPPS_Code': 'SBMHPSCD_unique',
        'Age': 'Age_mean',
        'Agency_Medicare_Number': 'M0010_counts'
    })
    
    # Normalize County to Title Case to match selected_county from grouped_df
    grouped['County'] = grouped['County'].astype(str).str.title()
    # Also ensure the input df has a temporary normalized column for joining if needed, 
    # but we are iterating using the grouped values so we need to match back to df.
    # If df['COUNTY_NAME'] is UPPER, and we changed grouped['County'] to Title, 
    # then iterating "for county in grouped['County']" and checking "df['COUNTY_NAME'] == county" will FAIL.
    
    # So we must fix the casing for the iteration or create a mapping.
    # accurate:
    # df['COUNTY_NAME'] values are distinct keys.
    # We iterated grouped['County'] which are now Title Case.
    # df['COUNTY_NAME'] is still original.
    
    # Better approach: Create a temp column in df for matching
    df = df.copy() # Avoid SettingWithCopyWarning on the subset
    df['County_Normalized'] = df['COUNTY_NAME'].astype(str).str.title()
    
    # Now use County_Normalized for matching inside the loops

    # Get spatial constants
    # Update to use the normalized key if possible, or just merge carefully.
    spatial_data = df[['County_Normalized'] + spatial_cols].drop_duplicates('County_Normalized')
    spatial_data = spatial_data.rename(columns={'County_Normalized': 'County'})
    grouped = grouped.merge(spatial_data, on='County', how='left')
    
    # Process categorical counts (Gender, Race, Discipline, BMI)
    # 1. Gender
    # Gender
    level_col = 'County_Normalized' # Use this for iteration
    
    # Gender
    gender_list = []
    for county in grouped['County']:
        sub = df[df[level_col] == county]
        counts = sub['Gender'].value_counts().to_dict()
        gender_list.append(counts)
    grouped['Gender_counts'] = gender_list

    # Race - Columns are: American_Indian_or_Alaska_Native, Asian, Black_or_African_American, Hispanic_or_Latino, Native_Hawiian_or_Pacific_Islander, White
    # These seem to be boolean or 1/0? Need to check. Assuming they are numeric 1/0.
    race_cols = ['American_Indian_or_Alaska_Native', 'Asian', 'Black_or_African_American', 
                 'Hispanic_or_Latino', 'Native_Hawiian_or_Pacific_Islander', 'White']
                 
    # Ensure numeric
    for col in race_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    race_list = []
    for county in grouped['County']:
        sub = df[df[level_col] == county]
        # Sum the columns
        raw_counts = sub[race_cols].sum()
        
        # Condense categories
        condensed_counts = {
            'White': raw_counts.get('White', 0),
            'Black': raw_counts.get('Black_or_African_American', 0),
            'Hispanic': raw_counts.get('Hispanic_or_Latino', 0),
            'Asian': raw_counts.get('Asian', 0),
            'Other': (raw_counts.get('American_Indian_or_Alaska_Native', 0) + 
                      raw_counts.get('Native_Hawiian_or_Pacific_Islander', 0))
        }
        race_list.append(condensed_counts)
    grouped['Race_counts'] = race_list

    # Discipline
    disc_list = []
    for county in grouped['County']:
        sub = df[df[level_col] == county]
        counts = sub['ByDiscipline'].value_counts().to_dict()
        disc_list.append(counts)
    grouped['ByDiscipline_counts'] = disc_list

    # BMI Category
    bmi_list = []
    for county in grouped['County']:
        sub = df[df[level_col] == county]
        counts = sub['BMI_Category'].value_counts().to_dict()
        bmi_list.append(counts)
    grouped['BMI_Category_counts'] = bmi_list

    # Add HHA_AGNCY_ID_unique placeholder (not in agg_funcs above but used in stats_text)
    # stats_text uses HHA_AGNCY_ID_unique. In grouped_df it exists. 
    # In icdData? 'Agency_Medicare_Number' is likely it. 
    grouped['HHA_AGNCY_ID_unique'] = grouped['M0010_counts']

    return grouped



def extract_and_plot(county_df, column_name, title):
    """ Extracts category counts and returns a Plotly bar plot. """

    statistic_df = extract_category_counts(county_df, column_name)

    category_counts = statistic_df.groupby('Category')['Count'].sum().reset_index()

    fig = px.bar(
        category_counts,
        x='Category',
        y='Count',
        title=title,
        labels={'Count': 'Population Count'},
        color='Category'
    )
    # Customize x-axis tick labels (make sure they fit and are readable)
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',  # Display labels in a linear fashion
            tickfont=dict(size=12),  # Font size for x-axis labels
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            title="Population Count",  # Set y-axis title
        ),
        title_x=0.5,  # Center title
        title_font=dict(size=14),  # Title font size
    )
    return fig
# Show Distribution and Map
def show_distribution_and_map(selected_county, df, texas_counties_geojson, condition_label="Diabetic"):
    county_df = df[df['County'] == selected_county]

    if county_df.empty:
        st.error(f"No data available for {selected_county}.")
        return [], None, None, ""

    # Category Mapping for Distribution
    categories = {
        'Gender_counts': 'Gender Distribution',
        'Race_counts': 'Race Distribution',
        'ByDiscipline_counts': 'Discipline Distribution',
        'BMI_Category_counts': 'BMI Category Distribution',
    }

    # Generate Distribution Plots
    plots = [extract_and_plot(county_df, col, title) for col, title in categories.items()]
    plots = [p for p in plots if p is not None]  # Remove None values

    # **Choropleth Map**
    fig_map = px.choropleth(
        county_df,
        geojson=texas_counties_geojson,
        locations='County',
        hover_data=['POP_URB', 'POPPCT_URB', 'POP_RUR', 'POPPCT_RUR'],
        featureidkey="properties.NAME",
        color='POPPCT_RUR',
        color_continuous_scale=px.colors.sequential.Viridis ,
        color_discrete_sequence=CHART_COLORS,
        title=f'Urbanization Map of {selected_county}'
    )
    fig_map.update_geos(fitbounds="locations", visible=False)

    # **Donut Chart for Urban vs Rural Population**
    urb_pct = county_df['POPPCT_URB'].values[0] if 'POPPCT_URB' in county_df.columns and not county_df.empty else 0
    rur_pct = county_df['POPPCT_RUR'].values[0] if 'POPPCT_RUR' in county_df.columns and not county_df.empty else 0

    fig_donut = go.Figure(go.Pie(
        labels=["Urban", "Rural"],
        values=[urb_pct, rur_pct],
        hole=0.4,
        marker=dict(colors=["#2ca02c", "#1f77b4"])# Green & Blue
    ))
    fig_donut.update_layout(title="Urban vs Rural Population %", font=dict(size=12))

    stats_dict = {
        "Beneficiaries": county_df['BENE_ID_unique'].values[0] if not county_df.empty else 0,
        "Facilities": county_df['FACINTID_unique'].values[0] if not county_df.empty else 0,
        "HHA Agencies": county_df['HHA_AGNCY_ID_unique'].values[0] if not county_df.empty else 0,
        "Avg Age": round(county_df['Age_mean'].values[0], 1) if not county_df.empty else 0
    }

    return plots, fig_map, fig_donut, stats_dict

def urban_rural_maps(df, selected_county):
    # **Urban Choropleth Map**
    fig_urban = px.choropleth(
        df,
        geojson=texas_counties_geojson,
        locations='County',
        hover_data=['County', 'POP_URB', 'POPPCT_URB'],
        featureidkey="properties.NAME",
        color='POPPCT_URB',
        color_continuous_scale=px.colors.sequential.Viridis  ,
        title='Urban County Population %',
    )
    fig_urban.update_geos(fitbounds="locations", visible=False)
    fig_urban.update_layout(
        font=dict(size=12)  # This sets the font size
    )
    # **Rural Choropleth Map**
    fig_rural = px.choropleth(
        df,
        geojson=texas_counties_geojson,
        locations='County',
        hover_data=[ 'County', 'POP_RUR', 'POPPCT_RUR'],
        featureidkey="properties.NAME",
        color='POPPCT_RUR',
        color_continuous_scale=px.colors.sequential.Viridis  ,
        title='Rural County Population %'
    )
    fig_rural.update_geos(fitbounds="locations", visible=False)
    fig_rural.update_layout(
        font=dict(size=12)  # This sets the font size
    )

    county_data = df[df['County'] == selected_county]
    if county_data.empty:
        # Fallback empty figure
        county_fig_rural = go.Figure()
        county_fig_rural.update_layout(title=f"No data for {selected_county}")
    else:
        county_fig_rural = px.choropleth(
            county_data,
            geojson=texas_counties_geojson,
            locations='County',
            hover_data=['County', 'POP_RUR', 'POPPCT_RUR', 'POP_URB', 'POPPCT_URB'],
            featureidkey="properties.NAME",
            color='POPPCT_RUR',
            color_continuous_scale=px.colors.sequential.Viridis  ,
            title=f'Rural Population % for {selected_county}'
        )
        county_fig_rural.update_geos(fitbounds="locations", visible=False)
        county_fig_rural.update_layout(
            font=dict(size=12)  # This sets the font size
        )

    return fig_urban, fig_rural, county_fig_rural

# **Header**
st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
        font-family: Helvetica Neue;
        font-size: 40px;
    }
    .header-container h1 {
        font-size: 40px;
        font-family: Helvetica Neue;
    }
    .header-container p {
        font-size: 18px;
        font-family: Helvetica Neue;
    }
    </style>
    <div class="header-container">
        <h1>PRECISION‑Connect</h1>
        <h1>Texan Population Health Analytics Platform</h1>
        <p>Explore demographic distributions and urbanization statistics across Texas counties.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# **Sidebar**
with st.sidebar:
    st.title('👩‍🏫 PRECISION‑Connect')
    
    # Debug info
    # if st.checkbox("Show Debug Info"):
    #     st.write("Columns:", icdData.columns.tolist())
    #     st.write("Data Types:", icdData.dtypes)
    #     st.write("Sample Data:", icdData.head())
    
    selected_county = st.selectbox("Select a County", sorted(grouped_df['County'].unique()))

# Tabs
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Diabetes", "Heart Failure", "Hypertension", "Custom Cohort"])

def render_tab_content(condition_name, df_subset, selected_county):
    if df_subset.empty:
        st.warning(f"No patients found for condition: {condition_name}")
        return

    # Aggregate data
    agg_df = aggregate_patient_data(df_subset)
    
    # Show Visualizations
    plots, fig_map, fig_donut, stats_dict = show_distribution_and_map(selected_county, agg_df, texas_counties_geojson, condition_name)
    
    # --- COMPACT LAYOUT FOR PRINT ---
    
    # 1. Headline & Stats
    st.subheader(f"{condition_name} Stats: {selected_county}")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1: st.metric("Beneficiaries", stats_dict["Beneficiaries"])
    with col_kpi2: st.metric("Facilities", stats_dict["Facilities"])
    with col_kpi3: st.metric("HHA Agencies", stats_dict["HHA Agencies"])
    with col_kpi4: st.metric("Avg Age", stats_dict["Avg Age"])
    
    st.write("---")

    # 2. Main Maps & Donut (Row 1)
    # Left: Main Map, Middle: Donut, Right: Urban/Rural Maps (Stacked?)
    
    # Let's try: 
    # Col 1 (Large): Urbanization Map 
    # Col 2 (Medium): Donut 
    # Col 3 (Medium): Urban/Rural reference
    
    c1, c2, c3 = st.columns([2, 1, 2])
    
    with c1:
        st.caption(f"Urbanization Map ({selected_county})")
        fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_map, use_container_width=True)
        
    with c2:
        st.caption("Pop. Distribution")
        fig_donut.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=200)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c3:
        fig_urban, fig_rural, county_fig_rural = urban_rural_maps(agg_df, selected_county)
        st.caption("State Ref: Rural %")
        fig_rural.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_rural, use_container_width=True)
        
    st.write("---")
    
    # 3. Demographics (Row 2 - Grid of 4)
    st.subheader("Demographics")
    g1, g2, g3, g4 = st.columns(4)
    
    # Update layout to be compact
    for p in plots:
        p.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=200, title_font=dict(size=12))

    with g1: 
        if len(plots) > 0: st.plotly_chart(plots[0], use_container_width=True)
    with g2: 
        if len(plots) > 1: st.plotly_chart(plots[1], use_container_width=True)
    with g3: 
        if len(plots) > 2: st.plotly_chart(plots[2], use_container_width=True)
    with g4: 
        if len(plots) > 3: st.plotly_chart(plots[3], use_container_width=True)


with tab1:
    subset = icdData[icdData['has_diabetes'] == 1]
    render_tab_content("Diabetes", subset, selected_county)

with tab2:
    subset = icdData[icdData['has_heart_failure'] == 1]
    render_tab_content("Heart Failure", subset, selected_county)

with tab3:
    subset = icdData[icdData['has_hypertension'] == 1]
    render_tab_content("Hypertension", subset, selected_county)

with tab4:
    st.subheader("Custom Comorbidity Filter")
    conditions = ['has_diabetes', 'has_heart_failure', 'has_hypertension']
    # Add other conditions if available in icdData
    extra_conditions = [c for c in ['aids', 'alcohol', 'depre', 'drug', 'obese', 'dementia'] if c in icdData.columns] 
    # Just using the ones requested + maybe risks? icdData columns shown in prompt:
    # 'has_diabetes', 'has_heart_failure', 'has_hypertension', 'ever_deceased', 'ever_readmitted'
    # Prompt showed: 'aids', 'alcohol', 'chf' (in risk scores?).
    # icdData columns: 'has_diabetes', 'has_heart_failure', 'has_hypertension'. 
    # I'll stick to these 3 for now in multiselect, user can request more.
    selected_conditions = st.multiselect("Select Conditions", conditions, default=conditions[:1])
    
    if selected_conditions:
        # Filter: Patients having ANY of the selected conditions? Or ALL?
        # Usually user meaning "Filter for patients that have A and B". (Intersection)
        # But "Diabetes OR Heart Failure" is also valid.
        # Let's assume Intersection (ALL selected) for specific cohort.
        mask = pd.Series(True, index=icdData.index)
        for cond in selected_conditions:
            mask = mask & (icdData[cond] == 1)
        
        subset = icdData[mask]
        render_tab_content("Custom Selection", subset, selected_county)




def images():
    im1 = os.path.join(current_path, "figs/txst_cads_logo.png")
    im2 = os.path.join(current_path, "figs/datalabimage.png")
    im3 = os.path.join(current_path, "figs/txstlogo.jpg")


    image1 = Image.open(im1)
    image2 = Image.open(im2)
    image3 = Image.open(im3)
    #image4 = Image.open('path_to_image1.jpg')
    image_height = 200
    image_width = 300

    # Create a white background of the desired size
    def add_background(image, width, height):
        new_image = Image.new("RGB", (width, height), (255, 255, 255))  # White background
        # Resize the original image
        resized_image = image.resize((width, height))
        # Paste the resized image onto the white background
        new_image.paste(resized_image, (0, 0))
        return new_image

    image1 = add_background(image1, image_width, image_height)
    return image1, image2, image3

with st.sidebar:
    #st.plotly_chart(fig_map, use_container_width=True)
    #st.plotly_chart(fig_donut, use_container_width=True)
    #st.write("\n" * 10)
    #st.image(image1, use_column_width=True)
    #st.image(image2,use_column_width=True)
    #st.image(image3,use_column_width=True)
    # About section
    with st.expander("About"):
        st.markdown("""
        ### Precision-Connect Interactive System
        This interactive system provides insights into the health and demographic data of Texas counties, focusing on key factors such as urbanization, health disparities, and more.

        **Key Features:**
        - **Demographic Distributions:** View the distribution of various demographics across Texas counties.
        - **Urban vs Rural Population:** See the breakdown of urban and rural populations.
        - **Health Insights:** Get insights on health-related statistics and visualize key health indicators for different counties.

        **Created by:**
        - Mirna Elizondo (m_e172@txstate.edu)
        - [Datalab12](https://datalab12.github.io/)
        - Texas State Center for Analytics and Data Science [TXST CADS](https://cads.txst.edu/)
        - [St. David's School of Nursing](https://www.nursing.txst.edu/)
        """)
    with st.expander("Data Sources"):
        st.markdown("""
        **Data Sources:**
        - The data used in this dashboard is derived from publicly available sources, including state health departments and national demographic surveys.

        1. **CMS OASIS Data**
           - **Description**: Contains detailed information on Texan home health patients, including demographics, diagnoses, care plans, and outcomes.
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

st.write("-------------")

topics = ['Attainment', 'Internet Connectivity', 'Living Conditions', 'Poverty','English Fluency']

def plot_heatmap(selected_topic, df):
    df = df[['County', 'POPPCT_RUR', 'POPPCT_URB']]
    merged_df = disparity_df.merge(df, right_on='County', left_on='County')
    # Classify counties as Rural or Urban
    merged_df["Classification"] = merged_df.apply(
        lambda row: "Rural" if row["POPPCT_RUR"] >= row["POPPCT_URB"] else "Urban", axis=1
    )

    # Filter the DataFrame based on the selected topic
    filtered_df = merged_df[merged_df["Topic"] == selected_topic]

    if filtered_df.empty:
        return "No data available for this selection."

    # Separate data for Rural and Urban
    rural_df = filtered_df[filtered_df["Classification"] == "Rural"]
    urban_df = filtered_df[filtered_df["Classification"] == "Urban"]

    # Create pivot tables
    rural_pivot = rural_df.pivot(index="County", columns="Indicator", values="AbsoluteIndexDisparity").fillna(0)
    urban_pivot = urban_df.pivot(index="County", columns="Indicator", values="AbsoluteIndexDisparity").fillna(0)

    # Create heatmaps
    fig_rural = px.imshow(rural_pivot.T, color_continuous_scale='Viridis', title="Rural Heatmap")
    fig_urban = px.imshow(urban_pivot.T, color_continuous_scale='Viridis', title="Urban Heatmap")

    # Adjust layout size
    fig_rural.update_layout(height=400, width=400, font_size=12)
    fig_urban.update_layout(height=400, width=400, font_size=12)

    return fig_rural, fig_urban

def plot_treemap(selected_county, selected_topic):
    # Filter the DataFrame based on selected county and topic
    filtered_df = disparity_df[(disparity_df["County"] == selected_county) & (disparity_df["Topic"] == selected_topic)]

    if filtered_df.empty:
        return "No data available for this selection."

    # List of disparity measures to create treemaps for
    disparity_measures = ['AbsoluteIndexDisparity', 'RelativeIndexDisparity',
                          'PopulationWeightedIndexDisparity']

    fig_list = []  # List to store the generated treemaps

    for measure in disparity_measures:
        # Create a treemap for each disparity measure
        fig = px.treemap(
            filtered_df,
            path=['Indicator'],
            values=measure,
            color='Indicator',
            color_discrete_sequence=CHART_COLORS,
            title=f"Treemap of {measure} ({selected_topic})"
        )
        fig.update_layout(font_size=12)
        fig_list.append(fig)
          # Append the figure to the list

    return fig_list


def plot_dendrogram(selected_county, selected_topic):
    # Filter the dataframe for selected county and topic
    filtered_df = disparity_df[(disparity_df["County"] == selected_county) & (disparity_df["Topic"] == selected_topic)]

    if filtered_df.empty:
        return "No data available for this selection."

    # List of disparity measures to visualize
    disparity_measures = ['AbsoluteIndexDisparity', 'RelativeIndexDisparity',
                          'PopulationWeightedIndexDisparity']

    # Pivot the dataframe to include all disparity measures as columns
    pivot_df = filtered_df.pivot(index='Indicator', columns='Topic', values=disparity_measures).fillna(0)

    # Cluster the disparity measures using hierarchical clustering
    try:
        # Use the complete linkage method to perform hierarchical clustering
        linkage_matrix = sch.linkage(pivot_df, method='ward')
    except ValueError as e:
        return "Error: *Not enough data*"

    # Plot the dendrogram
    plt.figure(figsize=(30, 8))  # Adjusted the figure size for dendrogram
    sch.dendrogram(linkage_matrix, labels=pivot_df.index, leaf_rotation=90, leaf_font_size=10)
    plt.title(f"Dendrogram of Indicators - {selected_county} ({selected_topic})")
    plt.xlabel("Indicators")
    plt.ylabel("Cluster Distance")
    plt.show()

    return plt


def create_visualizations(selected_county, grouped_df):
    topic_panels = {}

    for topic in topics:

        fig_rural, fig_urban = plot_heatmap(topic, grouped_df)
        treemaps = plot_treemap(selected_county, topic)
        dendrogram_panel = plot_dendrogram(selected_county, topic)

        topic_panels[topic] = (fig_rural, fig_urban, treemaps, dendrogram_panel)

    return topic_panels

topics = {
    'Attainment': {
        'columns': {
            'ACS_PCT_BACHELOR_DGR': 'Percentage of the population with a Bachelor\'s degree.',
            'ACS_PCT_COLLEGE_ASSOCIATE_DGR': 'Percentage of the population with an Associate\'s degree.',
            'ACS_PCT_GRADUATE_DGR': 'Percentage of the population with a graduate degree (Master\'s or higher).',
            'ACS_PCT_HS_GRADUATE': 'Percentage of the population with a high school diploma or equivalent.',
            'ACS_PCT_LT_HS': 'Percentage of the population without a high school diploma.',
            'ACS_PCT_NO_WORK_NO_SCHL_16_19': 'Percentage of youth (ages 16-19) neither working nor attending school.',
            'ACS_PCT_POSTHS_ED': 'Percentage of the population with post-high school education (including some college).',
            'ACS_PCT_VET_BACHELOR': 'Percentage of veterans with a Bachelor\'s degree.',
            'ACS_PCT_VET_COLLEGE': 'Percentage of veterans with a college degree (Associate\'s or higher).',
            'ACS_PCT_VET_HS': 'Percentage of veterans with a high school diploma or equivalent.',
        },
        '# Columns': 10,
        'Description': 'Measures the educational level of the population, including percentages of high school, associate, bachelor’s, and graduate degrees, as well as individuals with post-high school education.'},
    'Internet Connectivity': {
        'columns': {
            'ACS_PCT_HH_BROADBAND': 'Percentage of households with broadband internet access.',
            'ACS_PCT_HH_BROADBAND_ONLY': 'Percentage of households with broadband internet access only (no cellular).',
            'ACS_PCT_HH_CELLULAR': 'Percentage of households with cellular data service.',
            'ACS_PCT_HH_CELLULAR_ONLY': 'Percentage of households that only have cellular service (no broadband).',
            'ACS_PCT_HH_DIAL_INTERNET_ONLY': 'Percentage of households with dial-up internet access only.',
            'ACS_PCT_HH_INTERNET': 'Percentage of households with any form of internet access.',
            'ACS_PCT_HH_INTERNET_NO_SUBS': 'Percentage of households with internet access but no subscription service.',
            'ACS_PCT_HH_NO_COMP_DEV': 'Percentage of households with no computer or device.',
            'ACS_PCT_HH_NO_INTERNET': 'Percentage of households with no internet access.',
            'ACS_PCT_HH_OTHER_COMP': 'Percentage of households with other types of computers or devices (not listed).',
            'ACS_PCT_HH_OTHER_COMP_ONLY': 'Percentage of households with only other types of computers or devices.',
            'ACS_PCT_HH_PC': 'Percentage of households with personal computers.',
            'ACS_PCT_HH_PC_ONLY': 'Percentage of households with personal computers but no other devices.',
            'ACS_PCT_HH_SAT_INTERNET': 'Percentage of households with satellite internet access.',
            'ACS_PCT_HH_SMARTPHONE': 'Percentage of households with smartphones.',
            'ACS_PCT_HH_SMARTPHONE_ONLY': 'Percentage of households that only have smartphones (no broadband).',
            'ACS_PCT_HH_TABLET': 'Percentage of households with tablet devices.',
            'ACS_PCT_HH_TABLET_ONLY': 'Percentage of households that only have tablets (no broadband).',
        },
        '# Columns': 19,
        'Description':'Assesses access to broadband, smartphone, tablet, and computer devices, including households with and without internet access.'
        },
    'Living Conditions': {
        'columns': {
            'ACS_PCT_CHILDREN_GRANDPARENT': 'Percentage of children living with grandparents who are the primary caregivers.',
            'ACS_PCT_CHILD_1FAM': 'Percentage of children living in single-parent families.',
            'ACS_PCT_GRANDP_NO_RESPS': 'Percentage of grandparents without any responsibility for grandchildren.',
            'ACS_PCT_GRANDP_RESPS_NO_P': 'Percentage of grandparents responsible for grandchildren but not living with them.',
            'ACS_PCT_GRANDP_RESPS_P': 'Percentage of grandparents responsible for grandchildren and living with them.',
            'ACS_PCT_HH_1PERS': 'Percentage of households with one person.',
            'ACS_PCT_HH_ABOVE65': 'Percentage of households with at least one person aged 65 or older.',
            'ACS_PCT_HH_ALONE_ABOVE65': 'Percentage of households with only one person aged 65 or older.',
            'ACS_PCT_HH_KID_1PRNT': 'Percentage of households with children living with only one parent.',
            'ACS_TOT_GRANDCHILDREN_GP': 'Total number of grandchildren living in households with grandparents as primary caregivers.',
        },
        '# Columns': 10,
        'Description': 'Highlights household structures, such as single-person households, families with children, and those with grandparents as caregivers.'
        },
    'Poverty': {
        'columns': {
            'ACS_PCT_HEALTH_INC_138_199': 'Percentage of individuals with income between 138% and 199% of the federal poverty level.',
            'ACS_PCT_HEALTH_INC_200_399': 'Percentage of individuals with income between 200% and 399% of the federal poverty level.',
            'ACS_PCT_HEALTH_INC_ABOVE400': 'Percentage of individuals with income above 400% of the federal poverty level.',
            'ACS_PCT_HEALTH_INC_BELOW137': 'Percentage of individuals with income below 137% of the federal poverty level.',
            'ACS_PCT_HH_1FAM_FOOD_STMP': 'Percentage of single-parent households receiving food stamps.',
            'ACS_PCT_HH_FOOD_STMP': 'Percentage of all households receiving food stamps.',
            'ACS_PCT_HH_FOOD_STMP_BLW_POV': 'Percentage of households receiving food stamps with income below the poverty level.',
            'ACS_PCT_HH_NO_FD_STMP_BLW_POV': 'Percentage of households below the poverty level without food stamp assistance.',
            'ACS_PCT_HH_PUB_ASSIST': 'Percentage of households receiving public assistance.',
            'ACS_PCT_INC50': 'Percentage of households with an income below $50,000.',
            'ACS_PCT_INC50_ABOVE65': 'Percentage of households with an income below $50,000 and a member over 65.',
            'ACS_PCT_NONVET_POV_18_64': 'Percentage of non-veterans aged 18-64 living below the poverty level.',
            'ACS_PCT_PERSON_INC_100_124': 'Percentage of individuals with an income between 100% and 124% of the federal poverty level.',
            'ACS_PCT_PERSON_INC_125_199': 'Percentage of individuals with an income between 125% and 199% of the federal poverty level.',
            'ACS_PCT_PERSON_INC_ABOVE200': 'Percentage of individuals with an income above 200% of the federal poverty level.',
            'ACS_PCT_PERSON_INC_BELOW99': 'Percentage of individuals with an income below 99% of the federal poverty level.',
            'ACS_PCT_POV_AIAN': 'Percentage of American Indian/Alaska Native individuals living below the poverty level.',
            'ACS_PCT_POV_ASIAN': 'Percentage of Asian individuals living below the poverty level.',
            'ACS_PCT_POV_BLACK': 'Percentage of Black or African American individuals living below the poverty level.',
            'ACS_PCT_POV_HISPANIC': 'Percentage of Hispanic or Latino individuals living below the poverty level.',
            'ACS_PCT_POV_MULTI': 'Percentage of individuals of mixed races living below the poverty level.',
            'ACS_PCT_POV_NHPI': 'Percentage of Native Hawaiian/Pacific Islander individuals living below the poverty level.',
            'ACS_PCT_POV_OTHER': 'Percentage of individuals of other races living below the poverty level.',
            'ACS_PCT_POV_WHITE': 'Percentage of White individuals living below the poverty level.',
            'ACS_PCT_VET_POV_18_64': 'Percentage of veterans aged 18-64 living below the poverty level.',
            'ACS_TOT_CIVIL_NONINST_POP_POV': 'Total number of individuals in the civilian noninstitutional population living below the poverty level.',
            'ACS_TOT_CIVIL_POP_POV': 'Total number of individuals in the civilian population living below the poverty level.',
            'ACS_TOT_POP_POV': 'Total number of individuals in the total population living below the poverty level.',
        },
        '# Columns': 28,
        'Description':'Tracks economic status, including poverty rates, income levels, food assistance, and public assistance programs across different demographics.'},
    'English Fluency': {
        'columns': {
            'ACS_PCT_HH_LIMIT_ENGLISH': 'Percentage of households with limited English proficiency.',
        },
        '# Columns': 1,
        'Description': 'Indicates the percentage of households with limited English proficiency, affecting access to services and opportunities.', }
}

# Function to display the selected topic's columns and descriptions
def display_topic_info(selected_topic):
    for topic, details in topics.items():
        new_columns = {}
        for col, description in details['columns'].items():
            # Remove 'ACS_PCT' from the column name and replace underscores with spaces
            new_col_name = col.replace('ACS_PCT_', '').replace('_', ' ')
            new_columns[new_col_name] = description
        details['columns'] = new_columns
    topic_info = topics[selected_topic]
    # Show topic description
    st.write(f"**Number of columns for {selected_topic}:** {topic_info['# Columns']}")
    st.write("**Columns and description:**")
    for column, column_desc in topic_info['columns'].items():
        st.write(f"- **{column}**: {column_desc}")




# Streamlit interface
# SDOH Section - This remains common? Or should it be inside tabs too?
# User said "performs the current streamlit app...". 
# The SDOH part is part of the app. It uses disparity_df which is not filtered by patients.
# So we can keep it at the bottom, or duplicate it.
# Keeping it at the bottom makes sense as "Community Disparity Factors".

st.write("-------------")
st.title("Disparity Analysis Dashboard")
selected_topic = st.radio("Select a Topic", topics)

with st.expander("Description"):
    display_topic_info(selected_topic)

# SDOH visualizations don't depend on patient selection (they use disparity_df), so we pass grouped_df (which has static rural/urban info)
# However, create_visualizations uses detailed disparity_df.
topic_panels = create_visualizations(selected_county, grouped_df)
fig_rural, fig_urban, treemaps, dendrogram = topic_panels[selected_topic]

st.plotly_chart(fig_rural, use_container_width=True)
#st.plotly_chart(fig_urban, use_container_width=True)
#st.plotly_chart(fig_urban, use_container_width=True)
# Treemaps Display
if isinstance(treemaps, str):
    st.markdown(treemaps)
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(treemaps[0], use_container_width=True)
    with col2:
        st.plotly_chart(treemaps[1], use_container_width=True)
    with col3:
        st.plotly_chart(treemaps[2], use_container_width=True)


st.write("-------------")
