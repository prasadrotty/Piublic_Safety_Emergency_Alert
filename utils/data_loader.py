import pandas as pd
import streamlit as st

# ==========================================
# DATA LOADER
# ==========================================

@st.cache_data
def load_data():

    try:

        df = pd.read_csv(
            "data/Emergency_Alert_Dataset.csv"
        )

        return df

    except Exception as e:

        st.error(
            f"Error loading dataset: {e}"
        )

        return pd.DataFrame()

# ==========================================
# DATA SUMMARY
# ==========================================

def get_dataset_summary(df):

    summary = {

        "Total Rows":
            df.shape[0],

        "Total Columns":
            df.shape[1],

        "Missing Values":
            df.isnull().sum().sum(),

        "Duplicate Rows":
            df.duplicated().sum()
    }

    return summary

# ==========================================
# NUMERIC FEATURES
# ==========================================

def get_numeric_columns(df):

    return df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

# ==========================================
# CATEGORICAL FEATURES
# ==========================================

def get_categorical_columns(df):

    return df.select_dtypes(
        include=["object"]
    ).columns.tolist()

# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):

    return df.drop_duplicates()

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

def fill_missing_values(df):

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
# RISK SCORE
# ==========================================

def calculate_risk_score(df):

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
# DATA PREPROCESSING
# ==========================================

def preprocess_data():

    df = load_data()

    df = remove_duplicates(df)

    df = fill_missing_values(df)

    df = calculate_risk_score(df)

    return df

# ==========================================
# FILTER DATA
# ==========================================

def filter_data(
    df,
    incident_type=None,
    severity=None
):

    if incident_type:

        df = df[
            df["incident_type"].isin(
                incident_type
            )
        ]

    if severity:

        df = df[
            df["emergency_severity"].isin(
                severity
            )
        ]

    return df

# ==========================================
# TOP RISK ALERTS
# ==========================================

def get_top_risk_alerts(
    df,
    top_n=10
):

    if "risk_score" in df.columns:

        return df.sort_values(
            by="risk_score",
            ascending=False
        ).head(top_n)

    return df.head(top_n)

# ==========================================
# DATA HEALTH REPORT
# ==========================================

def data_health_report(df):

    report = pd.DataFrame({

        "Column":
            df.columns,

        "Missing Values":
            df.isnull().sum().values,

        "Data Type":
            df.dtypes.values

    })

    return report
