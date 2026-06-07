import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Risk Analysis",
    page_icon="⚠",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/Emergency_Alert_Dataset.csv"
    )

df = load_data()

# =====================================
# HEADER
# =====================================

st.title("⚠ Emergency Risk Analysis")

st.markdown("""
Advanced Risk Assessment System for
Emergency Alert Intelligence Platform
""")

st.divider()

# =====================================
# SIDEBAR FILTERS
# =====================================

st.sidebar.header("Filters")

incident_filter = st.sidebar.multiselect(
    "Incident Type",
    df["incident_type"].unique(),
    default=df["incident_type"].unique()
)

severity_filter = st.sidebar.multiselect(
    "Severity",
    df["emergency_severity"].unique(),
    default=df["emergency_severity"].unique()
)

df = df[
    (df["incident_type"].isin(incident_filter))
    &
    (df["emergency_severity"].isin(severity_filter))
]

# =====================================
# RISK SCORE CALCULATION
# =====================================

df["risk_score"] = (

    df["temperature_celsius"] * 0.20 +

    df["smoke_density_ppm"] * 0.20 +

    df["fire_area_ratio"] * 100 * 0.20 +

    df["crowd_density"] * 100 * 0.15 +

    df["traffic_density"] * 100 * 0.15 +

    df["propagation_delay_sec"] * 0.10

)

# =====================================
# RISK CATEGORY
# =====================================

def classify_risk(score):

    if score < 50:
        return "Low"

    elif score < 100:
        return "Medium"

    elif score < 150:
        return "High"

    else:
        return "Critical"

df["risk_category"] = df[
    "risk_score"
].apply(classify_risk)

# =====================================
# KPI SECTION
# =====================================

avg_risk = round(
    df["risk_score"].mean(),
    2
)

max_risk = round(
    df["risk_score"].max(),
    2
)

critical_count = len(
    df[df["risk_category"] == "Critical"]
)

avg_delay = round(
    df["propagation_delay_sec"].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Risk",
    avg_risk
)

col2.metric(
    "Maximum Risk",
    max_risk
)

col3.metric(
    "Critical Cases",
    critical_count
)

col4.metric(
    "Avg Delay",
    avg_delay
)

st.divider()

# =====================================
# RISK GAUGE
# =====================================

st.subheader("🎯 Average Risk Meter")

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        title={
            "text":"Average Risk Score"
        },
        gauge={
            "axis":{
                "range":[0,200]
            }
        }
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# RISK DISTRIBUTION
# =====================================

st.subheader("📈 Risk Distribution")

fig = px.histogram(
    df,
    x="risk_score",
    nbins=40,
    color="risk_category",
    title="Risk Score Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# RISK CATEGORY PIE
# =====================================

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        df,
        names="risk_category",
        title="Risk Categories"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.bar(
        df["risk_category"]
        .value_counts()
        .reset_index(),
        x="risk_category",
        y="count",
        title="Risk Category Counts"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# SEVERITY VS RISK
# =====================================

st.subheader("🚨 Severity vs Risk")

fig = px.box(
    df,
    x="emergency_severity",
    y="risk_score",
    color="emergency_severity",
    title="Risk by Severity"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# INCIDENT TYPE RISK
# =====================================

st.subheader("🔥 Incident Type Risk Analysis")

incident_risk = (
    df.groupby(
        "incident_type"
    )["risk_score"]
    .mean()
    .reset_index()
)

fig = px.bar(
    incident_risk,
    x="incident_type",
    y="risk_score",
    color="risk_score",
    title="Average Risk by Incident"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# SENSOR RISK ANALYSIS
# =====================================

st.subheader("🌡 Sensor Risk Relationship")

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        df,
        x="temperature_celsius",
        y="risk_score",
        color="risk_category",
        title="Temperature vs Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="smoke_density_ppm",
        y="risk_score",
        color="risk_category",
        title="Smoke Density vs Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# CORRELATION HEATMAP
# =====================================

st.subheader("📊 Risk Correlation Heatmap")

numeric_cols = df.select_dtypes(
    include=np.number
)

corr = numeric_cols.corr()

fig = px.imshow(
    corr,
    aspect="auto",
    title="Feature Correlation Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# TOP RISK ALERTS
# =====================================

st.subheader("🚨 Top 20 High-Risk Alerts")

top_risk = df.sort_values(
    "risk_score",
    ascending=False
).head(20)

columns_to_show = [

    "alert_id",
    "incident_type",
    "emergency_severity",
    "risk_score",
    "coverage_percentage",
    "response_time_sec"

]

st.dataframe(
    top_risk[columns_to_show],
    use_container_width=True
)

# =====================================
# AI INSIGHTS
# =====================================

st.subheader("🤖 AI Risk Insights")

highest_temp = round(
    df["temperature_celsius"].max(),
    2
)

highest_smoke = round(
    df["smoke_density_ppm"].max(),
    2
)

avg_coverage = round(
    df["coverage_percentage"].mean(),
    2
)

st.success(f"""

🔥 Maximum Temperature:
{highest_temp} °C

🌫 Maximum Smoke Density:
{highest_smoke}

📶 Average Coverage:
{avg_coverage} %

⚠ High Risk incidents are strongly associated
with elevated smoke density and temperature.

⚠ Propagation delays increase operational risk.

⚠ Critical alerts require immediate escalation.

🤖 AI recommends proactive monitoring
for incidents crossing the High Risk threshold.

""")

# =====================================
# RAW DATA
# =====================================

with st.expander("📄 View Risk Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Emergency Alert Intelligence Platform | Risk Analysis Module"
)
