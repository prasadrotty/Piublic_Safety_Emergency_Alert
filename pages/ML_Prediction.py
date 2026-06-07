import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="ML Prediction",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    feature_columns = joblib.load(
        "models/feature_columns.pkl"
    )

    return (
        model,
        scaler,
        label_encoder,
        feature_columns
    )

(
    model,
    scaler,
    label_encoder,
    feature_columns
) = load_artifacts()

# ==========================================
# TITLE
# ==========================================

st.title("🤖 Emergency Alert Prediction System")

st.markdown("""
Predict emergency alert severity using
sensor, propagation, environmental,
and network parameters.
""")

st.divider()

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("📥 Enter Alert Details")

col1, col2, col3 = st.columns(3)

with col1:

    temperature_celsius = st.number_input(
        "Temperature (°C)",
        0.0,
        100.0,
        35.0
    )

    smoke_density_ppm = st.number_input(
        "Smoke Density (PPM)",
        0.0,
        1000.0,
        150.0
    )

    co2_level_ppm = st.number_input(
        "CO₂ Level (PPM)",
        0.0,
        5000.0,
        500.0
    )

    humidity_percentage = st.slider(
        "Humidity %",
        0,
        100,
        50
    )

with col2:

    visibility_score = st.slider(
        "Visibility Score",
        0,
        100,
        80
    )

    fire_area_ratio = st.slider(
        "Fire Area Ratio",
        0.0,
        1.0,
        0.2
    )

    crowd_density = st.slider(
        "Crowd Density",
        0.0,
        1.0,
        0.4
    )

    traffic_density = st.slider(
        "Traffic Density",
        0.0,
        1.0,
        0.3
    )

with col3:

    propagation_delay_sec = st.number_input(
        "Propagation Delay",
        0.0,
        100.0,
        10.0
    )

    coverage_percentage = st.slider(
        "Coverage %",
        0,
        100,
        75
    )

    alert_reliability_score = st.slider(
        "Reliability Score",
        0.0,
        1.0,
        0.80
    )

    response_time_sec = st.number_input(
        "Response Time",
        0.0,
        500.0,
        50.0
    )

# ==========================================
# OPTIONAL CATEGORICALS
# ==========================================

st.subheader("📡 Additional Parameters")

incident_type = st.selectbox(
    "Incident Type",
    [
        "fire",
        "earthquake",
        "flood",
        "accident",
        "storm"
    ]
)

emergency_severity = st.selectbox(
    "Current Severity Indicator",
    [
        "low",
        "medium",
        "high",
        "critical"
    ]
)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "🚨 Predict Alert Risk",
    use_container_width=True
):

    try:

        # ==========================
        # CREATE INPUT DATAFRAME
        # ==========================

        input_data = {

            "temperature_celsius":
                temperature_celsius,

            "smoke_density_ppm":
                smoke_density_ppm,

            "co2_level_ppm":
                co2_level_ppm,

            "humidity_percentage":
                humidity_percentage,

            "visibility_score":
                visibility_score,

            "fire_area_ratio":
                fire_area_ratio,

            "crowd_density":
                crowd_density,

            "traffic_density":
                traffic_density,

            "propagation_delay_sec":
                propagation_delay_sec,

            "coverage_percentage":
                coverage_percentage,

            "alert_reliability_score":
                alert_reliability_score,

            "response_time_sec":
                response_time_sec,

            "incident_type":
                incident_type,

            "emergency_severity":
                emergency_severity
        }

        input_df = pd.DataFrame(
            [input_data]
        )

        # ==========================
        # DUMMY ENCODING
        # ==========================

        input_df = pd.get_dummies(
            input_df
        )

        # ==========================
        # MATCH TRAIN FEATURES
        # ==========================

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ==========================
        # SCALE
        # ==========================

        input_scaled = scaler.transform(
            input_df
        )

        # ==========================
        # PREDICT
        # ==========================

        prediction = model.predict(
            input_scaled
        )

        prediction_class = (
            label_encoder
            .inverse_transform(
                prediction
            )[0]
        )

        # ==========================
        # PROBABILITY
        # ==========================

        confidence = (
            np.max(
                model.predict_proba(
                    input_scaled
                )
            ) * 100
        )

        st.success(
            f"Prediction : {prediction_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # ==========================
        # RISK SCORE
        # ==========================

        risk_score = (

            temperature_celsius * 0.20 +

            smoke_density_ppm * 0.20 +

            fire_area_ratio * 100 * 0.20 +

            crowd_density * 100 * 0.15 +

            traffic_density * 100 * 0.15 +

            propagation_delay_sec * 0.10
        )

        st.metric(
            "Calculated Risk Score",
            round(risk_score, 2)
        )

        # ==========================
        # RECOMMENDATIONS
        # ==========================

        st.subheader(
            "🧠 Recommended Action"
        )

        prediction_text = str(
            prediction_class
        ).lower()

        if "critical" in prediction_text:

            st.error("""
            Immediate evacuation required.

            Dispatch emergency response teams.

            Notify authorities instantly.

            Activate highest alert protocol.
            """)

        elif "high" in prediction_text:

            st.warning("""
            High risk detected.

            Prepare emergency units.

            Monitor alert propagation.

            Increase surveillance.
            """)

        elif "medium" in prediction_text:

            st.info("""
            Moderate emergency.

            Continue monitoring.

            Keep response teams ready.
            """)

        else:

            st.success("""
            Situation stable.

            Routine monitoring recommended.
            """)

        # ==========================
        # FEATURE SUMMARY
        # ==========================

        st.subheader(
            "📋 Input Summary"
        )

        st.dataframe(
            pd.DataFrame(
                [input_data]
            ),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
"""
Emergency Alert Intelligence Platform
Machine Learning Prediction Module
"""
)
