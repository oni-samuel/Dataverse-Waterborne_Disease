import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load models
regression_model = xgb.XGBRegressor()
regression_model.load_model("xgboost_model.json")

classification_model = joblib.load("best_clas_model.pkl")

# Constants
START_YEAR = 2015
START_MONTH = 1

# Expected column order for regression model
FEATURE_ORDER = [
    'Turbidity(NTU)', 'Ecoli_Count(CFU/100ml)', 'Nitrate(mg/L)', 'pH',
    'Year', 'Month', 'Quarter', 'Time_Since_Start',
    'Community_Ajegunle', 'Community_Bagamoyo', 'Community_Bonny', 'Community_Chibombo',
    'Community_Dori', 'Community_Entebbe', 'Community_Garissa', 'Community_Gboko',
    'Community_Ikorodu', 'Community_Kasoa', 'Community_Kibera', 'Community_Lamu',
    'Community_Lokoja', 'Community_Makoko', 'Community_Maradi', 'Community_Mathare',
    'Community_Nsawam', 'Community_Nzega', 'Community_Takoradi', 'Community_Zinder',
    'Region_Coastal', 'Region_Dryland', 'Region_Peri-Urban', 'Region_Rural', 'Region_Urban Slum',
    'Season_Dry', 'Season_Rainy'
]

# UI
st.title("💧 Waterborne Disease Prediction App")

mode = st.sidebar.selectbox("Choose Prediction Type", ["Risk Level", "Waterborne Cases"])

year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2023)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=0)
quarter = ((month - 1) // 3) + 1
time_since_start = (year - START_YEAR) * 12 + (month - START_MONTH)

ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
turbidity = st.sidebar.number_input("Turbidity (NTU)", min_value=0.0, value=1.0, step=0.1)
ecoli_count = st.sidebar.number_input("Ecoli Count (CFU/100ml)", min_value=0, value=0, step=1)
nitrate = st.sidebar.number_input("Nitrate Concentration (mg/L)", min_value=0.0, value=1.0, step=0.1)

communities = [
    "Ajegunle", "Bagamoyo", "Bonny", "Chibombo", "Dori", "Entebbe", "Garissa", "Gboko", "Ikorodu", "Kasoa",
    "Kibera", "Lamu", "Lokoja", "Makoko", "Maradi", "Mathare", "Nsawam", "Nzega", "Takoradi", "Zinder"
]
selected_community = st.sidebar.selectbox("Community", communities)

regions = ["Coastal", "Dryland", "Peri-Urban", "Rural", "Urban Slum"]
selected_region = st.sidebar.selectbox("Region", regions)

seasons = ["Dry", "Rainy"]
selected_season = st.sidebar.selectbox("Season", seasons)

# Prediction
if st.button("Predict"):

    if mode == "Waterborne Cases":
        # Build regression input dict
        input_data = {
            "Turbidity(NTU)": [turbidity],
            "Ecoli_Count(CFU/100ml)": [ecoli_count],
            "Nitrate(mg/L)": [nitrate],
            "pH": [ph],
            "Year": [year],
            "Month": [month],
            "Quarter": [quarter],
            "Time_Since_Start": [time_since_start],
        }

        # One-hot encode
        for comm in communities:
            input_data[f"Community_{comm}"] = [1 if comm == selected_community else 0]
        for reg in regions:
            input_data[f"Region_{reg}"] = [1 if reg == selected_region else 0]
        for seas in seasons:
            input_data[f"Season_{seas}"] = [1 if seas == selected_season else 0]

        input_df = pd.DataFrame(input_data)

        # Ensure all expected columns exist
        for col in FEATURE_ORDER:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training exactly
        input_df = input_df[FEATURE_ORDER]

        total_cases = regression_model.predict(input_df)[0]
        st.success(f"Estimated Total Waterborne Cases: {round(total_cases):,}")

    else:
        # Classification expects raw columns
        input_df = pd.DataFrame({
            "Region": [selected_region],
            "Community": [selected_community],
            "Turbidity(NTU)": [turbidity],
            "Ecoli_Count(CFU/100ml)": [ecoli_count],
            "Nitrate(mg/L)": [nitrate],
            "pH": [ph],
            "Year": [year],
            "Month": [month],
            "Quarter": [quarter],
            "Time_Since_Start": [time_since_start],
            "Season": [selected_season],
        })

        risk_level_encoded = classification_model.predict(input_df)[0]
        risk_probas = classification_model.predict_proba(input_df)[0]

        risk_map = {0: "Low", 1: "Medium", 2: "High"}
        max_index = risk_probas.argmax()
        max_risk = risk_map.get(max_index, "Unknown")
        max_prob = risk_probas[max_index]

        st.success(f"Predicted Risk Level: {max_risk} ({max_prob:.2%} confidence)")
