import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Alert Analytics",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("data/Emergency_Alert_Dataset.csv")

df = load_data()

# ==========================================
# TITLE
# ==========================================

st.title("📊 Emergency Alert Analytics Dashboard")

st.markdown(
"""
Advanced analytics for emergency incidents,
sensor readings, alert reliability, and propagation metrics.
"""
)

st.divider()

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Filters")

incident = st.sidebar.multiselect(
    "Incident Type",
    options=df["incident_type"].unique(),
    default=df["incident_type"].unique()
)

severity = st.sidebar.multiselect(
    "Severity",
    options=df["emergency_severity"].unique(),
    default=df["emergency_severity"].unique()
)

df = df[
    (df["incident_type"].isin(incident))
    &
    (df["emergency_severity"].isin(severity))
]

# ==========================================
# KPI SECTION
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Alerts",
    f"{len(df):,}"
)

col2.metric(
    "Avg Temperature",
    round(df["temperature_celsius"].mean(), 2)
)

col3.metric(
    "Avg Smoke Density",
    round(df["smoke_density_ppm"].mean(), 2)
)

col4.metric(
    "Avg Reliability",
    round(df["alert_reliability_score"].mean(), 2)
)

st.divider()

# ==========================================
# TEMPERATURE ANALYSIS
# ==========================================

st.subheader("🌡 Temperature Analytics")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="temperature_celsius",
        nbins=30,
        color="emergency_severity",
        title="Temperature Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        x="emergency_severity",
        y="temperature_celsius",
        color="emergency_severity",
        title="Temperature by Severity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# SMOKE ANALYTICS
# ==========================================

st.subheader("🌫 Smoke Density Analytics")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="smoke_density_ppm",
        nbins=40,
        color="incident_type",
        title="Smoke Density Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="temperature_celsius",
        y="smoke_density_ppm",
        color="emergency_severity",
        title="Temperature vs Smoke Density"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# CO2 ANALYSIS
# ==========================================

st.subheader("🧪 CO₂ Analytics")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="co2_level_ppm",
        nbins=40,
        title="CO₂ Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="co2_level_ppm",
        y="visibility_score",
        color="incident_type",
        title="CO₂ vs Visibility"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# FIRE ANALYSIS
# ==========================================

st.subheader("🔥 Fire Area Analytics")

fig = px.box(
    df,
    x="incident_type",
    y="fire_area_ratio",
    color="incident_type",
    title="Fire Area Ratio by Incident Type"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# RESPONSE ANALYTICS
# ==========================================

st.subheader("🚑 Emergency Response Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="response_time_sec",
        nbins=30,
        color="emergency_severity",
        title="Response Time Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        x="emergency_severity",
        y="response_time_sec",
        color="emergency_severity",
        title="Response Time by Severity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# RELIABILITY ANALYSIS
# ==========================================

st.subheader("📡 Alert Reliability Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="alert_reliability_score",
        nbins=30,
        title="Reliability Score Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="alert_reliability_score",
        y="coverage_percentage",
        color="emergency_severity",
        title="Reliability vs Coverage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# SEVERITY ANALYSIS
# ==========================================

st.subheader("🚨 Severity Analytics")

severity_counts = (
    df["emergency_severity"]
    .value_counts()
    .reset_index()
)

severity_counts.columns = [
    "Severity",
    "Count"
]

fig = px.bar(
    severity_counts,
    x="Severity",
    y="Count",
    color="Severity",
    title="Emergency Severity Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# RISK SCORE
# ==========================================

st.subheader("⚠ Risk Score Analysis")

df["risk_score"] = (
    df["temperature_celsius"] * 0.20 +
    df["smoke_density_ppm"] * 0.20 +
    df["fire_area_ratio"] * 100 * 0.20 +
    df["crowd_density"] * 100 * 0.15 +
    df["traffic_density"] * 100 * 0.15 +
    df["propagation_delay_sec"] * 0.10
)

fig = px.histogram(
    df,
    x="risk_score",
    nbins=40,
    title="Risk Score Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.subheader("📈 Correlation Heatmap")

numeric_df = df.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    aspect="auto",
    title="Correlation Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# TOP RISK ALERTS
# ==========================================

st.subheader("🔥 Top 20 Highest Risk Alerts")

top_alerts = df.sort_values(
    "risk_score",
    ascending=False
).head(20)

st.dataframe(
    top_alerts,
    use_container_width=True
)

# ==========================================
# AI INSIGHTS
# ==========================================

st.subheader("🤖 AI Insights")

highest_temp = round(
    df["temperature_celsius"].max(),
    2
)

highest_smoke = round(
    df["smoke_density_ppm"].max(),
    2
)

avg_response = round(
    df["response_time_sec"].mean(),
    2
)

avg_coverage = round(
    df["coverage_percentage"].mean(),
    2
)

st.success(f"""
🔥 Highest Temperature Recorded: {highest_temp} °C

🌫 Highest Smoke Density: {highest_smoke}

🚑 Average Response Time: {avg_response} sec

📶 Average Alert Coverage: {avg_coverage} %

⚠ High smoke density strongly correlates with critical alerts.

⚠ Increased propagation delay can reduce alert effectiveness.

🤖 Monitoring sensor health and propagation networks improves emergency response quality.
""")

# ==========================================
# RAW DATA
# ==========================================

with st.expander("📄 View Full Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Emergency Alert Intelligence Platform | Alert Analytics Module"
)
