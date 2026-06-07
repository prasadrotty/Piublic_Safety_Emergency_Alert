import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Overview",
    page_icon="🚨",
    layout="wide"
)

# ====================================
# LOAD DATA
# ====================================

@st.cache_data
def load_data():
    return pd.read_csv("Emergency_Alert_Dataset.csv")

df = load_data()

# ====================================
# HEADER
# ====================================

st.title("🚨 Emergency Alert Intelligence Platform")
st.markdown("### System Overview Dashboard")

st.markdown("---")

# ====================================
# KPI SECTION
# ====================================

total_alerts = len(df)

critical_alerts = len(
    df[df["emergency_severity"] == "critical"]
)

avg_response = round(
    df["response_time_sec"].mean(), 2
)

avg_reliability = round(
    df["alert_reliability_score"].mean(), 2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Alerts",
    f"{total_alerts:,}"
)

col2.metric(
    "Critical Alerts",
    critical_alerts
)

col3.metric(
    "Avg Response Time",
    avg_response
)

col4.metric(
    "Reliability Score",
    avg_reliability
)

st.markdown("---")

# ====================================
# INCIDENT DISTRIBUTION
# ====================================

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        df,
        names="incident_type",
        title="Incident Type Distribution"
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
        title="Emergency Severity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ====================================
# SENSOR OVERVIEW
# ====================================

st.subheader("📡 Sensor Overview")

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        df,
        y="temperature_celsius",
        title="Temperature Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        y="smoke_density_ppm",
        title="Smoke Density Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ====================================
# ALERT PROPAGATION
# ====================================

st.subheader("🌐 Alert Propagation Metrics")

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        df,
        x="propagation_delay_sec",
        y="coverage_percentage",
        color="emergency_severity",
        title="Propagation Delay vs Coverage"
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

# ====================================
# RESPONSE ANALYTICS
# ====================================

st.subheader("🚑 Emergency Response Analytics")

col1, col2 = st.columns(2)

with col1:

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

with col2:

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

# ====================================
# COVERAGE ANALYSIS
# ====================================

st.subheader("📶 Coverage Analytics")

coverage = round(
    df["coverage_percentage"].mean(),
    2
)

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=coverage,
        title={"text": "Average Alert Coverage (%)"},
        gauge={
            "axis": {"range": [0, 100]}
        }
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ====================================
# TOP INCIDENTS
# ====================================

st.subheader("🔥 Top High-Risk Alerts")

risk_score = (
    df["temperature_celsius"] * 0.20 +
    df["smoke_density_ppm"] * 0.20 +
    df["crowd_density"] * 100 * 0.15 +
    df["traffic_density"] * 100 * 0.15 +
    df["fire_area_ratio"] * 100 * 0.20 +
    df["propagation_delay_sec"] * 0.10
)

df["risk_score"] = risk_score

top_alerts = df.sort_values(
    "risk_score",
    ascending=False
).head(10)

display_cols = [
    "alert_id",
    "incident_type",
    "emergency_severity",
    "risk_score",
    "response_time_sec",
    "coverage_percentage"
]

st.dataframe(
    top_alerts[display_cols],
    use_container_width=True
)

# ====================================
# AI INSIGHTS
# ====================================

st.subheader("🤖 AI Insights")

highest_temp = round(
    df["temperature_celsius"].max(),
    2
)

avg_delay = round(
    df["propagation_delay_sec"].mean(),
    2
)

avg_coverage = round(
    df["coverage_percentage"].mean(),
    2
)

avg_smoke = round(
    df["smoke_density_ppm"].mean(),
    2
)

st.success(f"""
🔥 Highest Recorded Temperature : {highest_temp} °C

🌫 Average Smoke Density : {avg_smoke}

🌐 Average Propagation Delay : {avg_delay} sec

📶 Average Coverage : {avg_coverage} %

🚨 Critical alerts require immediate intervention.

🤖 System indicates strong propagation-performance monitoring capability.
""")

# ====================================
# RAW DATA
# ====================================

with st.expander("📄 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# ====================================
# FOOTER
# ====================================

st.markdown("---")

st.caption(
    "Emergency Alert Intelligence Platform | Streamlit + Machine Learning + Social-Sensor Analytics"
)
