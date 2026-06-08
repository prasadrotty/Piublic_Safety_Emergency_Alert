import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title=" welcome to prasad Emergency Alert Intelligence Platform",
    page_icon="🚨",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("Emergency_Alert_Dataset.csv")
    return df

df = load_data()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🚨 Emergency Alert AI")

incident_filter = st.sidebar.multiselect(
    "Incident Type",
    options=df["incident_type"].unique(),
    default=df["incident_type"].unique()
)

severity_filter = st.sidebar.multiselect(
    "Emergency Severity",
    options=df["emergency_severity"].unique(),
    default=df["emergency_severity"].unique()
)

df = df[
    (df["incident_type"].isin(incident_filter)) &
    (df["emergency_severity"].isin(severity_filter))
]

# ==========================
# HEADER
# ==========================

st.title("🚨 Social-Sensor Emergency Alert Intelligence Platform")

st.markdown("""
Real-time analytics platform for emergency detection,
propagation analysis, severity prediction and operational insights.
""")

# ==========================
# KPI SECTION
# ==========================

total_alerts = len(df)

critical_alerts = len(
    df[df["emergency_severity"]=="critical"]
)

avg_response = round(
    df["response_time_sec"].mean(),2
)

avg_reliability = round(
    df["alert_reliability_score"].mean(),2
)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Alerts", total_alerts)
col2.metric("Critical Alerts", critical_alerts)
col3.metric("Avg Response Time", avg_response)
col4.metric("Reliability Score", avg_reliability)

st.divider()

# ==========================
# INCIDENT DISTRIBUTION
# ==========================

col1,col2 = st.columns(2)

with col1:

    fig = px.pie(
        df,
        names="incident_type",
        title="Incident Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        df,
        x="emergency_severity",
        color="emergency_severity",
        title="Severity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# SENSOR ANALYTICS
# ==========================

st.subheader("🔥 Sensor Analytics")

col1,col2 = st.columns(2)

with col1:

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

with col2:

    fig = px.scatter(
        df,
        x="co2_level_ppm",
        y="visibility_score",
        color="incident_type",
        title="CO2 vs Visibility"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# PROPAGATION ANALYSIS
# ==========================

st.subheader("🌐 Alert Propagation Analytics")

col1,col2 = st.columns(2)

with col1:

    fig = px.scatter(
        df,
        x="propagation_delay_sec",
        y="coverage_percentage",
        size="social_alerts_forwarded",
        color="emergency_severity",
        title="Delay vs Coverage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="node_influence_score",
        y="social_alerts_forwarded",
        color="incident_type",
        title="Node Influence Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# RELIABILITY ANALYSIS
# ==========================

st.subheader("📊 Reliability Analytics")

fig = px.histogram(
    df,
    x="alert_reliability_score",
    nbins=40,
    title="Alert Reliability Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# RISK SCORE
# ==========================

st.subheader("⚠ Risk Score Analysis")

df["risk_score"] = (

    df["temperature_celsius"]*0.20 +

    df["smoke_density_ppm"]*0.20 +

    df["fire_area_ratio"]*100*0.20 +

    df["crowd_density"]*100*0.15 +

    df["traffic_density"]*100*0.15 +

    df["propagation_delay_sec"]*0.10

)

fig = px.box(
    df,
    x="emergency_severity",
    y="risk_score",
    color="emergency_severity",
    title="Risk Score by Severity"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# CORRELATION HEATMAP
# ==========================

st.subheader("📈 Correlation Matrix")

numeric_cols = df.select_dtypes(
    include=np.number
)

corr = numeric_cols.corr()

fig = px.imshow(
    corr,
    aspect="auto",
    title="Feature Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# TOP ALERTS
# ==========================

st.subheader("🚨 Highest Risk Alerts")

top_alerts = df.sort_values(
    "risk_score",
    ascending=False
).head(10)

st.dataframe(
    top_alerts[
        [
            "alert_id",
            "incident_type",
            "emergency_severity",
            "risk_score",
            "coverage_percentage",
            "response_time_sec"
        ]
    ],
    use_container_width=True
)

# ==========================
# AI INSIGHTS
# ==========================

st.subheader("🤖 AI Generated Insights")

highest_temp = round(
    df["temperature_celsius"].max(),2
)

avg_delay = round(
    df["propagation_delay_sec"].mean(),2
)

best_reliability = round(
    df["alert_reliability_score"].max(),2
)

st.info(
f"""
🔥 Highest Temperature Detected:
{highest_temp} °C

🌐 Average Propagation Delay:
{avg_delay} sec

✅ Best Reliability Score:
{best_reliability}

📡 Average Coverage:
{round(df['coverage_percentage'].mean(),2)} %

🚑 Average Response Time:
{round(df['response_time_sec'].mean(),2)} sec
"""
)

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.markdown(
"""
Emergency Alert Intelligence Platform

Built with:
Python • Streamlit • Plotly • Machine Learning
"""
)
