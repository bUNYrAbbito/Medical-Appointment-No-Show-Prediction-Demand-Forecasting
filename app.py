import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Medical Appointment Analytics | CER Brazil",
    page_icon="🏥",
    layout="wide",
)

# =========================================================
# 2. ENCODING DICTIONARIES (Cleaned File ke exact mappings)
# =========================================================
SPECIALTY_MAP = {
    "Psychotherapy": 1,
    "Speech Therapy": 2,
    "Physiotherapy": 3,
    "Occupational Therapy": 4,
    "Assist": 5,
    "Pedagogo": 6,
    "Enf": 7,
    "Sem Especialidade": 8,
}

DISABILITY_MAP = {"Intellectual": 0, "Motor": 1}

RAIN_MAP = {"No Rain": 1, "Moderate": 2, "Weak": 3, "Heavy": 4}

HEAT_MAP = {"Mild": 1, "Cold": 2, "Warm": 3, "Heavy Cold": 4, "Heavy Warm": 5}

SHIFT_MAP = {"Afternoon": 1, "Morning": 2}

# =========================================================
# 3. DATA & MODEL LOADERS
# =========================================================


@st.cache_data
def load_data():
    # Use forward slash to prevent Windows '\a' path syntax errors
    df = pd.read_csv("clean/appointment_cleaned.csv")
    df["appointment_datetime"] = pd.to_datetime(
        df["appointment_datetime"], errors="coerce"
    )
    return df


@st.cache_resource
def load_models():
    no_show_package = joblib.load("models/best_no_show_model.pkl")
    demand_model = joblib.load("models/best_demand_forecasting_model.pkl")
    return no_show_package, demand_model


# Load artifacts
try:
    df = load_data()
    no_show_package, demand_model = load_models()
    no_show_model = no_show_package["model"]
    no_show_threshold = no_show_package.get("threshold", 0.45)
    no_show_features = no_show_package["features"]
except Exception as e:
    st.error(f"Error loading files/models: {e}")
    st.stop()


# =========================================================
# 4. FEATURE PREPARATION HELPER
# =========================================================
def prepare_no_show_input(user_data, training_cols):
    """Matches the exact feature transformation of training pipeline."""
    df_input = pd.DataFrame([user_data])

    dt = pd.to_datetime(df_input["appointment_datetime"].iloc[0])
    hour = dt.hour
    age = df_input["age"].iloc[0]

    # Derived temporal attributes
    df_input["appointment_day"] = dt.day
    df_input["appointment_month"] = dt.month
    df_input["appointment_dayofweek"] = dt.dayofweek
    df_input["appointment_hour"] = hour
    df_input["appointment_weekday"] = dt.day_name().lower()
    df_input["is_weekend"] = int(dt.dayofweek >= 5)

    # Time period
    if hour < 12:
        df_input["appointment_period"] = "morning"
    elif hour < 17:
        df_input["appointment_period"] = "afternoon"
    else:
        df_input["appointment_period"] = "evening"

    # Age group
    if age < 18:
        df_input["age_group"] = "under_18"
    elif age < 30:
        df_input["age_group"] = "18_29"
    elif age < 45:
        df_input["age_group"] = "30_44"
    elif age < 60:
        df_input["age_group"] = "45_59"
    else:
        df_input["age_group"] = "60_plus"

    # Encode remaining string categories with dummies (same as Notebook 2)
    categorical_cols = [
        "appointment_shift",
        "rain_intensity",
        "heat_intensity",
        "appointment_weekday",
        "appointment_period",
        "age_group",
    ]
    # Dummies create cols only if they are object type
    df_input = pd.get_dummies(
        df_input, columns=categorical_cols, drop_first=True, dtype=int
    )

    # Remove datetime object column
    if "appointment_datetime" in df_input.columns:
        df_input = df_input.drop(columns=["appointment_datetime"])

    # Align exactly with trained feature order (missing columns filled with 0)
    aligned_input = df_input.reindex(columns=training_cols, fill_value=0).astype(
        float
    )
    return aligned_input


# =========================================================
# 5. DASHBOARD TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    [
        "🚨 No-Show Prediction",
        "📈 Demand Forecasting",
        "📊 Clinic Business Analytics",
    ]
)

# ---------------------------------------------------------
# TAB 1: NO-SHOW PREDICTION (Future Dates Supported)
# ---------------------------------------------------------
with tab1:
    st.header("🚨 Patient No-Show Risk Assessment")
    st.write(
        "Enter appointment details to predict whether the patient is likely to miss their scheduled visit."
    )

    with st.form("no_show_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Patient Info")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input(
                "Age (Years)", min_value=0, max_value=110, value=32
            )
            disability = st.selectbox("Disability Type", list(DISABILITY_MAP.keys()))
            scholarship = st.selectbox("Bolsa Família Scholarship", ["No", "Yes"])

        with col2:
            st.subheader("📅 Schedule & Center")
            specialty_name = st.selectbox(
                "Specialty Required", list(SPECIALTY_MAP.keys())
            )
            app_date = st.date_input("Scheduled Date")
            app_time = st.time_input("Scheduled Time")
            shift_name = st.selectbox(
                "Appointment Shift", list(SHIFT_MAP.keys())
            )
            sms_received = st.selectbox("SMS Reminder Sent?", ["Yes", "No"])

        with col3:
            st.subheader("🩺 Health & Environment")
            hypertension = st.selectbox("Hypertension Condition", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            alcoholism = st.selectbox("Alcoholism History", ["No", "Yes"])
            rain_cond = st.selectbox(
                "Expected Rain Intensity", list(RAIN_MAP.keys())
            )
            heat_cond = st.selectbox(
                "Expected Heat Intensity", list(HEAT_MAP.keys())
            )

        submit_btn = st.form_submit_button(
            "Predict Attendance Risk", type="primary", use_container_width=True
        )

    if submit_btn:
        with st.spinner("Evaluating risk score..."):
            combined_dt = pd.Timestamp.combine(app_date, app_time)

            # Build record adhering to cleaned dataset encoding
            user_features = {
                "gender": 1 if gender == "Male" else 0,
                "age": age,
                "specialty": SPECIALTY_MAP[specialty_name],
                "disability": DISABILITY_MAP[disability],
                "appointment_shift": SHIFT_MAP[shift_name],
                "rain_intensity": RAIN_MAP[rain_cond],
                "heat_intensity": HEAT_MAP[heat_cond],
                "Hipertension": 1 if hypertension == "Yes" else 0,
                "Diabetes": 1 if diabetes == "Yes" else 0,
                "Alcoholism": 1 if alcoholism == "Yes" else 0,
                "Handcap": 1 if disability != "None" else 0,
                "Scholarship": 1 if scholarship == "Yes" else 0,
                "SMS_received": 1 if sms_received == "Yes" else 0,
                "appointment_datetime": combined_dt,
                # Impute typical weather medians from data
                "average_rain_day": float(df["average_rain_day"].median()),
                "max_temp_day": float(df["max_temp_day"].median()),
                "max_rain_day": float(df["max_rain_day"].median()),
                "average_temp_day": float(df["average_temp_day"].median()),
            }

            model_input = prepare_no_show_input(user_features, no_show_features)
            probability = no_show_model.predict_proba(model_input)[0][1]
            is_no_show = probability >= no_show_threshold

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Risk Score", f"{probability * 100:.2f}%")

            with c2:
                if is_no_show:
                    st.error("⚠️ HIGH RISK OF NO-SHOW DETECTED")
                    st.markdown(
                        """
                    **Recommended Immediate Actions:**
                    - Send a direct 2-way confirmation WhatsApp/SMS.
                    - Offer flexible rescheduling slots.
                    - Mark appointment slot for standby/waitlist backup.
                    """
                    )
                else:
                    st.success("✅ LOW RISK - HIGH PROBABILITY OF ATTENDANCE")
                    st.caption(
                        "Standard reminder schedule is adequate for this patient."
                    )


# ---------------------------------------------------------
# TAB 2: DEMAND FORECASTING (Time Series & Future Trend)
# ---------------------------------------------------------
with tab2:
    st.header("📈 Outpatient Appointment Demand Forecast")
    st.write(
        "Forecast clinic-wide volume to optimize medical staff schedules and specialist coverage."
    )

    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        lookback = st.slider("Lookback Historical Period (Days)", 14, 90, 30)
    with f_col2:
        forecast_days = st.slider("Forward Forecast Horizon (Days)", 1, 30, 7)

    # Historical Daily Aggregation
    daily_hist = (
        df.groupby(df["appointment_datetime"].dt.date)
        .size()
        .reset_index(name="actual_appointments")
    )
    daily_hist["appointment_datetime"] = pd.to_datetime(
        daily_hist["appointment_datetime"]
    )
    daily_hist = daily_hist.sort_values("appointment_datetime").set_index(
        "appointment_datetime"
    )

    st.subheader("Historical Volume")
    st.line_chart(daily_hist.tail(lookback))

    # Run inference with demand forecasting model
    if st.button("Generate Future Demand Forecast", type="primary"):
        with st.spinner("Computing time-series forecast..."):
            last_date = daily_hist.index.max()
            future_dates = [
                last_date + pd.Timedelta(days=i)
                for i in range(1, forecast_days + 1)
            ]

            future_df = pd.DataFrame({"forecast_date": future_dates})
            future_df["appointment_day"] = future_df["forecast_date"].dt.day
            future_df["appointment_month"] = future_df["forecast_date"].dt.month
            future_df["appointment_dayofweek"] = future_df[
                "forecast_date"
            ].dt.dayofweek
            future_df["is_weekend"] = (
                future_df["appointment_dayofweek"] >= 5
            ).astype(int)

            # Match model signature
            if hasattr(demand_model, "feature_names_in_"):
                aligned_demand_input = future_df.reindex(
                    columns=demand_model.feature_names_in_, fill_value=0
                )
            else:
                aligned_demand_input = future_df[
                    [
                        "appointment_day",
                        "appointment_month",
                        "appointment_dayofweek",
                        "is_weekend",
                    ]
                ]

            try:
                preds = demand_model.predict(aligned_demand_input)
                preds = np.clip(preds, a_min=0, a_max=None)  # No negative demand

                forecast_summary = pd.DataFrame(
                    {"Predicted Appointments": np.round(preds).astype(int)},
                    index=future_dates,
                )

                st.subheader(f"Forecast for Next {forecast_days} Days")
                st.line_chart(forecast_summary)
                st.dataframe(forecast_summary.T, use_container_width=True)
            except Exception as ex:
                st.warning(
                    f"Could not automatically parse future dimensions for regression model: {ex}"
                )


# ---------------------------------------------------------
# TAB 3: BUSINESS INSIGHTS & EDA SUMMARY
# ---------------------------------------------------------
with tab3:
    st.header("📊 Clinic Operations & Business Insights")

    total_records = len(df)
    no_show_total = (df["no_show"] == 1).sum()
    rate = (no_show_total / total_records) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Appointments Analyzed", f"{total_records:,}")
    m2.metric("Total Missed Visits", f"{no_show_total:,}")
    m3.metric("Baseline Clinic No-Show Rate", f"{rate:.2f}%")

    st.divider()

    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("No-Show Rate by Day of Week")
        weekday_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        df_weekday = (
            df.assign(weekday=df["appointment_datetime"].dt.day_name())
            .groupby("weekday")["no_show"]
            .mean()
            .mul(100)
            .reindex(weekday_names)
            .dropna()
        )
        st.bar_chart(df_weekday)

        st.subheader("No-Show Rate by Appointment Hour")
        df_hour = (
            df.assign(hour=df["appointment_datetime"].dt.hour)
            .groupby("hour")["no_show"]
            .mean()
            .mul(100)
        )
        st.line_chart(df_hour)

    with chart2:
        st.subheader("Attendance by Inverted Specialty Index")
        inv_map = {v: k for k, v in SPECIALTY_MAP.items()}
        spec_summary = (
            df.groupby("specialty")["no_show"]
            .mean()
            .mul(100)
            .rename(index=inv_map)
            .sort_values(ascending=False)
        )
        st.bar_chart(spec_summary)

        st.subheader("Key Business Takeaways")
        st.markdown(
            """
        1. **Overbooking Protocol**: Early morning sessions experience higher attendance volatility; consider low-percentage dynamic overbooking for high no-show specialties.
        2. **SMS Notification Timing**: Ensuring SMS reminders are dispatched 48h and 24h prior to appointments significantly drives down risk.
        3. **Weather Resilient Transport**: Patient attendance drops on high rain days; coordinate with community health transport on adverse weather days.
        """
        )

st.divider()
st.caption("CER Physical & Intellectual Rehabilitation Center • Production Deployment")