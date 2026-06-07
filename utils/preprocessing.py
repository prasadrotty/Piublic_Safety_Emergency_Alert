import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):

    return df.drop_duplicates()


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

def handle_missing_values(df):

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    return df


# ==========================================
# CREATE RISK SCORE
# ==========================================

def create_risk_score(df):

    required_cols = [

        "temperature_celsius",
        "smoke_density_ppm",
        "fire_area_ratio",
        "crowd_density",
        "traffic_density",
        "propagation_delay_sec"

    ]

    if all(
        col in df.columns
        for col in required_cols
    ):

        df["risk_score"] = (

            df["temperature_celsius"] * 0.20 +

            df["smoke_density_ppm"] * 0.20 +

            df["fire_area_ratio"] * 100 * 0.20 +

            df["crowd_density"] * 100 * 0.15 +

            df["traffic_density"] * 100 * 0.15 +

            df["propagation_delay_sec"] * 0.10

        )

    return df


# ==========================================
# CREATE PROPAGATION SCORE
# ==========================================

def create_propagation_score(df):

    required_cols = [

        "coverage_percentage",
        "social_alerts_forwarded",
        "node_influence_score",
        "alert_reliability_score"

    ]

    if all(
        col in df.columns
        for col in required_cols
    ):

        df["propagation_score"] = (

            df["coverage_percentage"] * 0.35 +

            df["social_alerts_forwarded"] * 0.25 +

            df["node_influence_score"] * 0.20 +

            df["alert_reliability_score"] * 100 * 0.20

        )

    return df


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def feature_engineering(df):

    df = create_risk_score(df)

    df = create_propagation_score(df)

    return df


# ==========================================
# LABEL ENCODING TARGET
# ==========================================

def encode_target(
    df,
    target_column
):

    label_encoder = LabelEncoder()

    df[target_column] = (
        label_encoder.fit_transform(
            df[target_column]
        )
    )

    return df, label_encoder


# ==========================================
# ONE HOT ENCODING
# ==========================================

def encode_features(df):

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df


# ==========================================
# SCALE FEATURES
# ==========================================

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ==========================================
# PREPARE TRAINING DATA
# ==========================================

def prepare_training_data(
    df,
    target_column
):

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = feature_engineering(df)

    y = df[target_column]

    X = df.drop(
        columns=[target_column]
    )

    X = encode_features(X)

    feature_columns = X.columns.tolist()

    X_scaled, scaler = scale_features(X)

    return (
        X_scaled,
        y,
        scaler,
        feature_columns
    )


# ==========================================
# PREPROCESS FOR PREDICTION
# ==========================================

def preprocess_prediction_input(
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
# DATA HEALTH REPORT
# ==========================================

def data_health_report(df):

    report = pd.DataFrame({

        "Column":
            df.columns,

        "Data Type":
            df.dtypes.values,

        "Missing Values":
            df.isnull().sum().values,

        "Unique Values":
            df.nunique().values

    })

    return report


# ==========================================
# COMPLETE PIPELINE
# ==========================================

def full_preprocessing_pipeline(df):

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = feature_engineering(df)

    return df
