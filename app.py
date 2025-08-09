import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb

# Load models
regression_model = xgb.XGBRegressor()
regression_model.load_model("xgboost_model.json")
classification_model = joblib.load("best_clas_model.pkl")

# Load encoders
encoders = joblib.load("label_encoders.pkl")
label_encoders = encoders['label_encoders']

# Lists of categorical and numerical features as per your dataset
categorical_features = ["Region", "Community", "Season"]
numerical_features = [
    "Turbidity(NTU)", "Ecoli_Count(CFU/100ml)", "Nitrate(mg/L)", "pH",
    "Year", "Month", "Quarter", "Time_Since_Start"
]

# For regression, one-hot encode categorical features manually in code
def onehot_encode_for_regression(input_dict):
    communities = [
        "Ajegunle", "Bagamoyo", "Bonny", "Chibombo", "Dori", "Entebbe", "Garissa", "Gboko",
        "Ikorodu", "Kasoa", "Kibera", "Lamu", "Lokoja", "Makoko", "Maradi", "Mathare",
        "Nsawam", "Nzega", "Takoradi", "Zinder"
    ]
    regions = ["Coastal", "Dryland", "Peri-Urban", "Rural", "Urban Slum"]
    seasons = ["Dry", "Rainy"]

    data = {k: [input_dict[k]] for k in numerical_features}

    # One-hot encode communities
    for comm in communities:
        data[f"Community_{comm}"] = [1 if input_dict["Community"] == comm else 0]
    # One-hot encode regions
    for reg in regions:
        data[f"Region_{reg}"] = [1 if input_dict["Region"] == reg else 0]
    # One-hot encode seasons
    for sea in seasons:
        data[f"Season_{sea}"] = [1 if input_dict["Season"] == sea else 0]

    df = pd.DataFrame(data)
    return df

def predict(input_dict):
    # Prepare classification input (label encode categorical)
    df_class = pd.DataFrame({})

    for col in categorical_features:
        df_class[col] = label_encoders[col].transform([input_dict[col]])
    for col in numerical_features:
        df_class[col] = [input_dict[col]]

    # Prepare regression input (one-hot encode categorical)
    df_reg = onehot_encode_for_regression(input_dict)

    # Predict regression (round to whole number)
    reg_result = int(round(float(regression_model.predict(df_reg)[0])))

    # Predict classification and probabilities
    risk_encoded = classification_model.predict(df_class)[0]
    risk_label = label_encoders['Risk_Level'].inverse_transform([risk_encoded])[0]

    if hasattr(classification_model, "predict_proba"):
        probas = classification_model.predict_proba(df_class)[0]
        max_idx = probas.argmax()
        max_label = label_encoders['Risk_Level'].inverse_transform([max_idx])[0]
        max_prob = probas[max_idx]
        prob_str = f"{max_label} ({max_prob:.2%})"
    else:
        prob_str = "N/A"

    return reg_result, risk_label, prob_str

# Streamlit UI
st.title("💧 Waterborne Disease Prediction App")

# Input fields
input_dict = {}

input_dict["Region"] = st.selectbox("Region", [
    "Coastal", "Dryland", "Peri-Urban", "Rural", "Urban Slum"
])

input_dict["Community"] = st.selectbox("Community", [
    "Ajegunle", "Bagamoyo", "Bonny", "Chibombo", "Dori", "Entebbe", "Garissa", "Gboko",
    "Ikorodu", "Kasoa", "Kibera", "Lamu", "Lokoja", "Makoko", "Maradi", "Mathare",
    "Nsawam", "Nzega", "Takoradi", "Zinder"
])

input_dict["Season"] = st.selectbox("Season", ["Dry", "Rainy"])

input_dict["Turbidity(NTU)"] = st.number_input("Turbidity (NTU)", min_value=0.0, value=1.0, step=0.1)
input_dict["Ecoli_Count(CFU/100ml)"] = st.number_input("E. coli Count (CFU/100ml)", min_value=0, value=0, step=1)
input_dict["Nitrate(mg/L)"] = st.number_input("Nitrate (mg/L)", min_value=0.0, value=1.0, step=0.1)
input_dict["pH"] = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
input_dict["Year"] = st.number_input("Year", min_value=2000, max_value=2100, value=2023, step=1)
input_dict["Month"] = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)
input_dict["Quarter"] = ((input_dict["Month"] - 1) // 3) + 1
input_dict["Time_Since_Start"] = (input_dict["Year"] - 2015) * 12 + (input_dict["Month"] - 1)

if st.button("Predict"):
    reg_pred, risk_pred, prob_pred = predict(input_dict)

    st.success(f"Estimated Total Waterborne Cases: {reg_pred:,}")
    st.success(f"Predicted Risk Level: {risk_pred}")
    st.write(f"Highest Probability Class: **{prob_pred}**")
