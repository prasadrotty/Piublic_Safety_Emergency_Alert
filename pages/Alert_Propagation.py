import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Alert Propagation",
    page_icon="🌐",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "Emergency_Alert_Dataset.csv"
    )

df = load_data()

# ==========================================
# HEADER
# ==========================================

st.title("🌐 Alert Propagation Intelligence")

st.markdown("""
Analyze how emergency alerts spread across
social networks, sensor networks, and response systems.
""")

st.divider()

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Propagation Filters")

incident_filter = st.sidebar.multiselect(
    "Incident Type",
    df["incident_type"].unique(),
    default=df["incident_type"].unique()
)

severity_filter = st.sidebar.multiselect(
    "Emergency Severity",
    df["emergency_severity"].unique(),
    default=df["emergency_severity"].unique()
)

df = df[
    (df["incident_type"].isin(incident_filter))
    &
    (df["emergency_severity"].isin(severity_filter))
]

# ==========================================
# KPI SECTION
# ==========================================

avg_delay = round(
    df["propagation_delay_sec"].mean(),
    2
)

avg_coverage = round(
    df["coverage_percentage"].mean(),
    2
)

avg_latency = round(
    df["network_latency_ms"].mean(),
    2
)

avg_forwarded = round(
    df["social_alerts_forwarded"].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Avg Propagation Delay",
    f"{avg_delay} sec"
)

col2.metric(
    "Avg Coverage",
    f"{avg_coverage}%"
)

col3.metric(
    "Avg Network Latency",
    f"{avg_latency} ms"
)

col4.metric(
    "Avg Alerts Forwarded",
    avg_forwarded
)

st.divider()

# ==========================================
# COVERAGE ANALYSIS
# ==========================================

st.subheader("📶 Coverage Analytics")

fig = px.histogram(
    df,
    x="coverage_percentage",
    nbins=30,
    color="emergency_severity",
    title="Coverage Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# PROPAGATION DELAY
# ==========================================

st.subheader("⏳ Propagation Delay Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="propagation_delay_sec",
        nbins=40,
        title="Propagation Delay Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        x="emergency_severity",
        y="propagation_delay_sec",
        color="emergency_severity",
        title="Propagation Delay by Severity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# DELAY VS COVERAGE
# ==========================================

st.subheader("📡 Delay vs Coverage")

fig = px.scatter(
    df,
    x="propagation_delay_sec",
    y="coverage_percentage",
    color="emergency_severity",
    size="social_alerts_forwarded",
    hover_data=["incident_type"],
    title="Propagation Delay vs Coverage"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# NETWORK LATENCY
# ==========================================

st.subheader("🌍 Network Latency Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="network_latency_ms",
        nbins=30,
        title="Network Latency Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.scatter(
        df,
        x="network_latency_ms",
        y="coverage_percentage",
        color="incident_type",
        title="Latency vs Coverage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# NODE INFLUENCE ANALYSIS
# ==========================================

st.subheader("🎯 Node Influence Analytics")

fig = px.scatter(
    df,
    x="node_influence_score",
    y="social_alerts_forwarded",
    color="emergency_severity",
    size="coverage_percentage",
    title="Node Influence vs Alert Forwarding"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# ALERT FORWARDING ANALYSIS
# ==========================================

st.subheader("📤 Social Alert Forwarding")

fig = px.box(
    df,
    x="incident_type",
    y="social_alerts_forwarded",
    color="incident_type",
    title="Forwarded Alerts by Incident Type"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# RELIABILITY ANALYSIS
# ==========================================

st.subheader("✅ Reliability vs Propagation")

fig = px.scatter(
    df,
    x="alert_reliability_score",
    y="coverage_percentage",
    color="emergency_severity",
    size="social_alerts_forwarded",
    title="Reliability vs Coverage"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# ROUTING PATH ANALYSIS
# ==========================================

if "routing_path_length" in df.columns:

    st.subheader("🛣 Routing Path Analysis")

    fig = px.histogram(
        df,
        x="routing_path_length",
        nbins=25,
        title="Routing Path Length Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# PROPAGATION SCORE
# ==========================================

st.subheader("🚨 Propagation Score Analysis")

df["propagation_score"] = (

    df["coverage_percentage"] * 0.35 +

    df["social_alerts_forwarded"] * 0.25 +

    df["node_influence_score"] * 0.20 +

    df["alert_reliability_score"] * 100 * 0.20

)

fig = px.histogram(
    df,
    x="propagation_score",
    nbins=35,
    title="Propagation Score Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# TOP PROPAGATED ALERTS
# ==========================================

st.subheader("🔥 Top 20 Propagated Alerts")

top_alerts = df.sort_values(
    "propagation_score",
    ascending=False
).head(20)

columns_to_show = [
    "alert_id",
    "incident_type",
    "emergency_severity",
    "coverage_percentage",
    "social_alerts_forwarded",
    "propagation_delay_sec"
]

available_columns = [
    col for col in columns_to_show
    if col in top_alerts.columns
]

st.dataframe(
    top_alerts[available_columns],
    use_container_width=True
)

# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.subheader("📊 Propagation Correlation Matrix")

numeric_df = df.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    aspect="auto",
    title="Propagation Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# AI INSIGHTS
# ==========================================

st.subheader("🤖 AI Propagation Insights")

best_coverage = round(
    df["coverage_percentage"].max(),
    2
)

best_influence = round(
    df["node_influence_score"].max(),
    2
)

lowest_delay = round(
    df["propagation_delay_sec"].min(),
    2
)

st.success(f"""
📶 Maximum Coverage Achieved:
{best_coverage} %

🎯 Highest Node Influence:
{best_influence}

⚡ Lowest Propagation Delay:
{lowest_delay} sec

📡 Alerts with high node influence
achieve greater dissemination.

📡 Lower propagation delay improves
coverage and emergency response.

📡 Reliable alerts are shared more
frequently across networks.

🤖 AI recommends prioritizing
high-influence nodes during emergency
communication campaigns.
""")

# ==========================================
# RAW DATA
# ==========================================

with st.expander("📄 View Propagation Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Emergency Alert Intelligence Platform | Alert Propagation Module"
)
