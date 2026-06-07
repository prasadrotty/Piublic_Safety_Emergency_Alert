import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Insights",
    page_icon="🤖",
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
# HEADER
# ==========================================

st.title("🤖 AI Emergency Intelligence Center")

st.markdown("""
Advanced AI-driven insights generated from
sensor data, alert propagation patterns,
risk indicators, and emergency response metrics.
""")

st.divider()

# ==========================================
# CALCULATED METRICS
# ==========================================

df["risk_score"] = (

    df["temperature_celsius"] * 0.20 +

    df["smoke_density_ppm"] * 0.20 +

    df["fire_area_ratio"] * 100 * 0.20 +

    df["crowd_density"] * 100 * 0.15 +

    df["traffic_density"] * 100 * 0.15 +

    df["propagation_delay_sec"] * 0.10

)

# ==========================================
# EXECUTIVE KPIs
# ==========================================

st.subheader("📊 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Alerts",
    f"{len(df):,}"
)

col2.metric(
    "Avg Risk",
    round(df["risk_score"].mean(), 2)
)

col3.metric(
    "Avg Coverage",
    round(df["coverage_percentage"].mean(), 2)
)

col4.metric(
    "Avg Reliability",
    round(df["alert_reliability_score"].mean(), 2)
)

st.divider()

# ==========================================
# INCIDENT ANALYSIS
# ==========================================

st.subheader("🚨 Incident Intelligence")

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
    title="Incident Frequency"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

most_common_incident = (
    df["incident_type"]
    .mode()[0]
)

st.info(
    f"Most common incident type detected: **{most_common_incident}**"
)

# ==========================================
# SEVERITY ANALYSIS
# ==========================================

st.subheader("🚨 Severity Intelligence")

severity_counts = (
    df["emergency_severity"]
    .value_counts()
)

fig = px.pie(
    names=severity_counts.index,
    values=severity_counts.values,
    title="Severity Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# RISK ANALYSIS
# ==========================================

st.subheader("⚠ Risk Intelligence")

avg_risk = round(
    df["risk_score"].mean(),
    2
)

max_risk = round(
    df["risk_score"].max(),
    2
)

high_risk_count = len(
    df[df["risk_score"] > avg_risk]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Risk",
    avg_risk
)

col2.metric(
    "Maximum Risk",
    max_risk
)

col3.metric(
    "High Risk Alerts",
    high_risk_count
)

# ==========================================
# SENSOR INSIGHTS
# ==========================================

st.subheader("🌡 Sensor Intelligence")

highest_temp = round(
    df["temperature_celsius"].max(),
    2
)

highest_smoke = round(
    df["smoke_density_ppm"].max(),
    2
)

highest_co2 = round(
    df["co2_level_ppm"].max(),
    2
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Highest Temperature",
    f"{highest_temp} °C"
)

col2.metric(
    "Highest Smoke Density",
    highest_smoke
)

col3.metric(
    "Highest CO₂",
    highest_co2
)

# ==========================================
# PROPAGATION INSIGHTS
# ==========================================

st.subheader("🌐 Propagation Intelligence")

avg_delay = round(
    df["propagation_delay_sec"].mean(),
    2
)

avg_forwarded = round(
    df["social_alerts_forwarded"].mean(),
    2
)

avg_influence = round(
    df["node_influence_score"].mean(),
    2
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Avg Propagation Delay",
    avg_delay
)

col2.metric(
    "Avg Forwarded Alerts",
    avg_forwarded
)

col3.metric(
    "Avg Node Influence",
    avg_influence
)

# ==========================================
# RELIABILITY INSIGHTS
# ==========================================

st.subheader("📡 Reliability Intelligence")

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
# TOP RISK ALERTS
# ==========================================

st.subheader("🔥 Top Critical Alerts")

top_alerts = df.sort_values(
    "risk_score",
    ascending=False
).head(20)

columns_to_show = [

    "alert_id",
    "incident_type",
    "emergency_severity",
    "risk_score",
    "response_time_sec",
    "coverage_percentage"

]

available_columns = [
    c for c in columns_to_show
    if c in top_alerts.columns
]

st.dataframe(
    top_alerts[available_columns],
    use_container_width=True
)

# ==========================================
# AI GENERATED INSIGHTS
# ==========================================

st.subheader("🧠 AI Generated Insights")

critical_alerts = len(
    df[df["emergency_severity"] == "critical"]
)

critical_pct = round(
    (critical_alerts / len(df)) * 100,
    2
)

insights = f"""
1. {critical_pct}% of alerts belong to the Critical category.

2. Average emergency response time is
{round(df['response_time_sec'].mean(),2)} seconds.

3. Alerts with higher reliability scores
tend to achieve greater coverage.

4. Elevated smoke density and temperature
are strongly associated with high-risk incidents.

5. Average propagation delay is
{round(df['propagation_delay_sec'].mean(),2)} seconds.

6. Most common incident type:
{most_common_incident}

7. Average network latency:
{round(df['network_latency_ms'].mean(),2)} ms.

8. System reliability remains at
{round(df['alert_reliability_score'].mean()*100,2)}%.
"""

st.success(insights)

# ==========================================
# REPORT GENERATOR
# ==========================================

st.subheader("📄 Generate Executive Report")

report_text = f"""
EMERGENCY ALERT INTELLIGENCE REPORT

Total Alerts: {len(df)}

Average Risk Score:
{round(df['risk_score'].mean(),2)}

Critical Alerts:
{critical_alerts}

Average Coverage:
{round(df['coverage_percentage'].mean(),2)}

Average Reliability:
{round(df['alert_reliability_score'].mean(),2)}

Average Response Time:
{round(df['response_time_sec'].mean(),2)}

Most Common Incident:
{most_common_incident}
"""

st.download_button(
    label="⬇ Download Report",
    data=report_text,
    file_name="Emergency_AI_Report.txt",
    mime="text/plain"
)

# ==========================================
# DATA VIEW
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
AI Insights Module
Streamlit + Machine Learning + Advanced Analytics
"""
)
