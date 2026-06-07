import pandas as pd
import numpy as np
import joblib

# ==========================================
# LOAD MODEL ARTIFACTS
# ==========================================

def load_model_artifacts():

    model = joblib.load(
        "models/model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

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

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

def create_input_dataframe(
    temperature_celsius,
    smoke_density_ppm,
    co2_level_ppm,
    humidity_percentage,
    visibility_score,
    fire_area_ratio,
    crowd_density,
    traffic_density,
    propagation_delay_sec,
    coverage_percentage,
    alert_reliability_score,
    response_time_sec,
    incident_type,
    emergency_severity
):

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

    return pd.DataFrame([input_data])

# ==========================================
# PREPROCESS INPUT
# ==========================================

def preprocess_input(
    input_df,
    feature_columns,
    scaler
):

    input_df = pd.get_dummies(
        input_df
    )

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(
        input_df
    )

    return input_scaled

# ==========================================
# PREDICT CLASS
# ==========================================

def predict_alert(
    input_scaled,
    model,
    label_encoder
):

    prediction = model.predict(
        input_scaled
    )

    predicted_class = (
        label_encoder
        .inverse_transform(
            prediction
        )[0]
    )

    confidence = float(

        np.max(
            model.predict_proba(
                input_scaled
            )
        ) * 100

    )

    return (
        predicted_class,
        confidence
    )

# ==========================================
# RISK SCORE
# ==========================================

def calculate_risk_score(
    temperature_celsius,
    smoke_density_ppm,
    fire_area_ratio,
    crowd_density,
    traffic_density,
    propagation_delay_sec
):

    risk_score = (

        temperature_celsius * 0.20 +

        smoke_density_ppm * 0.20 +

        fire_area_ratio * 100 * 0.20 +

        crowd_density * 100 * 0.15 +

        traffic_density * 100 * 0.15 +

        propagation_delay_sec * 0.10

    )

    return round(
        risk_score,
        2
    )

# ==========================================
# RISK CATEGORY
# ==========================================

def get_risk_category(
    risk_score
):

    if risk_score < 50:

        return "Low"

    elif risk_score < 100:

        return "Medium"

    elif risk_score < 150:

        return "High"

    else:

        return "Critical"

# ==========================================
# GENERATE RECOMMENDATION
# ==========================================

def generate_recommendation(
    prediction
):

    prediction = str(
        prediction
    ).lower()

    if "critical" in prediction:

        return """
Immediate evacuation required.

Deploy emergency response units.

Notify authorities immediately.

Activate highest alert protocol.
"""

    elif "high" in prediction:

        return """
High-risk emergency detected.

Prepare response teams.

Increase monitoring.

Enable rapid communication.
"""

    elif "medium" in prediction:

        return """
Moderate emergency detected.

Maintain active monitoring.

Keep emergency teams ready.
"""

    else:

        return """
Situation currently stable.

Routine monitoring recommended.
"""

# ==========================================
# FULL PREDICTION PIPELINE
# ==========================================

def run_prediction_pipeline(
    input_df
):

    (
        model,
        scaler,
        label_encoder,
        feature_columns
    ) = load_model_artifacts()

    input_scaled = preprocess_input(
        input_df,
        feature_columns,
        scaler
    )

    prediction, confidence = (
        predict_alert(
            input_scaled,
            model,
            label_encoder
        )
    )

    return (
        prediction,
        confidence
    )

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def get_feature_importance():

    try:

        model = joblib.load(
            "models/model.pkl"
        )

        feature_columns = joblib.load(
            "models/feature_columns.pkl"
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_df = pd.DataFrame({

                "Feature":
                    feature_columns,

                "Importance":
                    model.feature_importances_

            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )

            return importance_df

        return pd.DataFrame()

    except:

        return pd.DataFrame()

# ==========================================
# PREDICTION SUMMARY
# ==========================================

def generate_prediction_summary(
    prediction,
    confidence,
    risk_score
):

    summary = {

        "Predicted Severity":
            prediction,

        "Confidence":
            round(confidence, 2),

        "Risk Score":
            risk_score,

        "Risk Category":
            get_risk_category(
                risk_score
            )

    }

    return summary
