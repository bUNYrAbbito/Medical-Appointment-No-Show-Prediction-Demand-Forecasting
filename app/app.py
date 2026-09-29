import streamlit as st
import joblib
import pandas as pd
import os

st.title("Medical Appointment Prediction System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

clf_path = os.path.join(BASE_DIR, "..", "models", "no_show_model.pkl")
forecast_path = os.path.join(BASE_DIR, "..", "models", "demand_forecast_model.pkl")

clf = joblib.load(clf_path)
forecast = joblib.load(forecast_path)

tab1, tab2 = st.tabs(["No-Show Predictor", "Demand Forecast"])

# ---------------- NO SHOW ----------------
with tab1:
    st.header("No-Show Prediction")

    age = st.number_input("Age", 1, 100)
    sms = st.selectbox("SMS Received", [0,1])
    hyper = st.selectbox("Hypertension", [0,1])
    diab = st.selectbox("Diabetes", [0,1])

    day = st.slider("Day of Appointment", 1, 31)
    month = st.slider("Month", 1, 12)
    weekday = st.slider("Weekday (0=Mon)", 0, 6)

    if st.button("Predict No-Show"):

        df = pd.DataFrame([[age, sms, hyper, diab, day, month, weekday]],
            columns=[
                'age',
                'SMS_received',
                'Hipertension',
                'Diabetes',
                'day',
                'month',
                'weekday'
            ])

        pred = clf.predict(df)[0]

        if pred == 1:
            st.error("High Risk: Patient may MISS appointment")
        else:
            st.success("Low Risk: Patient likely to ATTEND")

# ---------------- FORECAST ----------------
with tab2:
    st.header("Demand Forecast")

    day = st.slider("Select Day", 1, 31)
    month = st.slider("Select Month", 1, 12)
    weekday = st.slider("Select Weekday (0=Mon, 6=Sun)", 0, 6)

    if st.button("Predict Demand"):

        input_df = pd.DataFrame(
            [[day, month, weekday]],
            columns=['day','month','weekday']
        )

        result = forecast.predict(input_df)[0]

        st.success(f"Expected Appointments: {int(result)}")
