import streamlit as st
import pandas as pd
import joblib

# Load saved models and encoders
regression_model = joblib.load("xgboost_model.pkl")
classification_model = joblib.load("best_clas_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

START_YEAR = 2015
START_MONTH = 1

st.title("💧 Waterborne Disease Prediction App")

# Sidebar: choose prediction type
mode = st.sidebar.selectbox("Choose Prediction Type", ["Risk Level", "Waterborne Cases"])

# Sidebar inputs
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2023, step=1)
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

if mode == "Waterborne Cases":
    # Regression model input: one-hot encoding for community, region, season
    input_data = {
        "Turbidity(NTU)": [turbidity],
        "Ecoli_Count(CFU/100ml)": [ecoli_count],
        "Nitrate(mg/L)": [nitrate],
        "pH": [ph],
        "Year": [year],
        "Month": [month],
        "Quarter": [quarter],
        "Time_Since_Start": [time_since_start]
    }

    for comm in communities:
        input_data[f"Community_{comm}"] = [1 if comm == selected_community else 0]

    for reg in regions:
        input_data[f"Region_{reg}"] = [1 if reg == selected_region else 0]

    for seas in seasons:
        input_data[f"Season_{seas}"] = [1 if seas == selected_season else 0]

    input_df = pd.DataFrame(input_data)

else:
    # Classification model input: label encode categorical features with correct order
    region_enc = label_encoders['Region'].transform([selected_region])[0]
    community_enc = label_encoders['Community'].transform([selected_community])[0]
    season_enc = label_encoders['Season'].transform([selected_season])[0]

    input_dict = {
        'Region': region_enc,
        'Community': community_enc,
        'Turbidity(NTU)': turbidity,
        'Ecoli_Count(CFU/100ml)': ecoli_count,
        'Nitrate(mg/L)': nitrate,
        'pH': ph,
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'Time_Since_Start': time_since_start,
        'Season': season_enc
    }

    expected_features = ['Region', 'Community', 'Turbidity(NTU)', 'Ecoli_Count(CFU/100ml)', 
                         'Nitrate(mg/L)', 'pH', 'Year', 'Month', 'Quarter', 'Time_Since_Start', 'Season']

    input_df = pd.DataFrame(columns=expected_features)
    for col in expected_features:
        input_df.at[0, col] = input_dict[col]

# Prediction button
if st.button("Predict"):
    if mode == "Waterborne Cases":
        total_cases = regression_model.predict(input_df)[0]
        st.success(f"Estimated Total Waterborne Cases: {round(total_cases):,}")
    else:
        risk_level_encoded = classification_model.predict(input_df)[0]
        risk_label = label_encoders['Risk_Level'].inverse_transform([risk_level_encoded])[0]
        st.success(f"Predicted Risk Level: {risk_label}")

        if hasattr(classification_model, "predict_proba"):
            probas = classification_model.predict_proba(input_df)[0]
            max_idx = probas.argmax()
            max_label = label_encoders['Risk_Level'].inverse_transform([max_idx])[0]
            max_prob = probas[max_idx]
            st.write(f"Predicted Risk Level Probability: **{max_label}** with probability **{max_prob:.2%}**")
