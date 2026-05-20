# ==============================
# Heart Disease Prediction App
# ==============================

import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Saved Model
# ==============================

model = joblib.load("heart_disease_model.pkl")

# ==============================
# App Title
# ==============================

st.title("Heart Disease Prediction System")

st.write("Enter patient information below:")

# ==============================
# User Inputs
# ==============================

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    [1, 2, 3, 4]
)

bp = st.number_input(
    "Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "FBS over 120",
    [0, 1]
)

ekg = st.selectbox(
    "EKG Results",
    [0, 1, 2]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=250,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    [0, 1]
)

st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope_st = st.selectbox(
    "Slope of ST",
    [1, 2, 3]
)

vessels = st.selectbox(
    "Number of Vessels Fluro",
    [0, 1, 2, 3]
)

thallium = st.selectbox(
    "Thallium",
    [3, 6, 7]
)

# ==============================
# Predict Button
# ==============================

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        'ID': 1,
        'Age': age,
        'Sex': sex,
        'Chest pain type': chest_pain,
        'BP': bp,
        'Cholesterol': cholesterol,
        'FBS over 120': fbs,
        'EKG results': ekg,
        'Max HR': max_hr,
        'Exercise angina': exercise_angina,
        'ST depression': st_depression,
        'Slope of ST': slope_st,
        'Number of vessels fluro': vessels,
        'Thallium': thallium
    }])

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)

    # Display result
    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    # Show probability
    st.write(
        "Prediction Probability:",
        round(probability[0][1] * 100, 2),
        "%"
    )