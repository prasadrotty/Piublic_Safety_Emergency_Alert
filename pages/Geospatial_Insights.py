import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Geospatial Insights",
    page_icon="🗺️",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/Emergency_Alert_Dataset.csv"
    )

df = load_data()

# ==========================================
# TITLE
# ==========================================

st.title("🗺️ Geospatial Emergency Intelligence")

st.markdown("""
Location-based analytics for emergency incidents,
risk hotspots, alert coverage, and response efficiency.
""")

st.divider()

# ==========================================
# DETECT LAT/LON COLUMNS
# ==========================================

lat_col = None
lon_col = None

possible_lat = [
    "latitude",
    "lat",
    "Latitude",
    "LATITUDE"
]

possible_lon = [
    "longitude",
    "lon",
    "lng",
    "Longitude",
    "LONGITUDE"
]

for col in df.columns:

    if col in possible_lat:
        lat_col = col

    if col in possible_lon:
        lon_col = col

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Location Filters")

if "incident_type" in df.columns:

    incident_filter = st.sidebar.multiselect(
        "Incident Type",
        options=df["incident_type"].unique(),
        default=df["incident_type"].unique()
    )

    df = df[
        df["incident_type"].isin(
            incident_filter
        )
    ]

if "emergency_severity" in df.columns:

    severity_filter = st.sidebar.multiselect(
        "Severity",
        options=df["emergency_severity"].unique(),
        default=df["emergency_severity"].unique()
    )

    df = df[
        df["emergency_severity"].isin(
            severity_filter
        )
    ]

# ==========================================
# KPI SECTION
# ==========================================

st.subheader("📊 Geographic KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Alerts",
    len(df)
)

if "coverage_percentage" in df.columns:

    col2.metric(
        "Avg Coverage",
        round(
            df["coverage_percentage"].mean(),
            2
        )
    )

if "response_time_sec" in df.columns:

    col3.metric(
        "Avg Response Time",
        round(
            df["response_time_sec"].mean(),
            2
        )
    )

if "alert_reliability_score" in df.columns:

    col4.metric(
        "Avg Reliability",
        round(
            df["alert_reliability_score"].mean(),
            2
        )
    )

st.divider()

# ==========================================
# MAP VISUALIZATION
# ==========================================

st.subheader("📍 Emergency Incident Map")

if lat_col and lon_col:

    map_df = df[
        [lat_col, lon_col]
    ].copy()

    st.map(map_df)

else:

    st.warning("""
    Latitude and Longitude columns
    not found in dataset.

    Map visualization unavailable.
    """)

# ==========================================
# RISK SCORE
# ==========================================

if all(
    col in df.columns
    for col in [
        "temperature_celsius",
        "smoke_density_ppm",
        "fire_area_ratio",
        "crowd_density",
        "traffic_density",
        "propagation_delay_sec"
    ]
):

    df["risk_score"] = (

        df["temperature_celsius"] * 0.20 +

        df["smoke_density_ppm"] * 0.20 +

        df["fire_area_ratio"] * 100 * 0.20 +

        df["crowd_density"] * 100 * 0.15 +

        df["traffic_density"] * 100 * 0.15 +

        df["propagation_delay_sec"] * 0.10

    )

# ==========================================
# INCIDENT DISTRIBUTION
# ==========================================

st.subheader("🚨 Incident Distribution")

if "incident_type" in df.columns:

    incident_counts = (
        df["incident_type"]
        .value_counts()
        .reset_index()
    )

    incident_counts.columns = [
        "Incident",
        "Count"
    ]

    fig = px.bar(
        incident_counts,
        x="Incident",
        y="Count",
        color="Count",
        title="Incidents by Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# SEVERITY ANALYSIS
# ==========================================

st.subheader("⚠ Severity Distribution")

if "emergency_severity" in df.columns:

    fig = px.pie(
        df,
        names="emergency_severity",
        title="Emergency Severity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# COVERAGE ANALYSIS
# ==========================================

if "coverage_percentage" in df.columns:

    st.subheader("📶 Coverage Analysis")

    fig = px.histogram(
        df,
        x="coverage_percentage",
        nbins=30,
        title="Coverage Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# RESPONSE ANALYSIS
# ==========================================

if "response_time_sec" in df.columns:

    st.subheader("🚑 Response Time Analysis")

    fig = px.histogram(
        df,
        x="response_time_sec",
        nbins=30,
        title="Response Time Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# RISK HOTSPOTS
# ==========================================

if "risk_score" in df.columns:

    st.subheader("🔥 Risk Hotspots")

    top_risk = df.sort_values(
        "risk_score",
        ascending=False
    ).head(20)

    cols = [
        c for c in [
            "alert_id",
            "incident_type",
            "emergency_severity",
            "risk_score"
        ]
        if c in top_risk.columns
    ]

    st.dataframe(
        top_risk[cols],
        use_container_width=True
    )

# ==========================================
# AI INSIGHTS
# ==========================================

st.subheader("🤖 Geographic AI Insights")

insights = []

if "coverage_percentage" in df.columns:

    insights.append(
        f"Average coverage is {round(df['coverage_percentage'].mean(),2)}%"
    )

if "response_time_sec" in df.columns:

    insights.append(
        f"Average response time is {round(df['response_time_sec'].mean(),2)} seconds"
    )

if "incident_type" in df.columns:

    insights.append(
        f"Most common incident is {df['incident_type'].mode()[0]}"
    )

if "emergency_severity" in df.columns:

    insights.append(
        f"Most common severity is {df['emergency_severity'].mode()[0]}"
    )

for item in insights:

    st.success(item)

# ==========================================
# RAW DATA
# ==========================================

with st.expander(
    "📄 View Complete Dataset"
):

    st.dataframe(
        df,
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    """
    Emergency Alert Intelligence Platform
    | Geospatial Insights Module
    """
)
