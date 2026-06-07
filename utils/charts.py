import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# ==========================================
# PIE CHART
# ==========================================

def create_pie_chart(
    df,
    column,
    title
):

    fig = px.pie(
        df,
        names=column,
        title=title
    )

    return fig


# ==========================================
# BAR CHART
# ==========================================

def create_bar_chart(
    df,
    x_col,
    y_col,
    title,
    color_col=None
):

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title
    )

    return fig


# ==========================================
# HISTOGRAM
# ==========================================

def create_histogram(
    df,
    column,
    title,
    bins=30,
    color=None
):

    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        color=color,
        title=title
    )

    return fig


# ==========================================
# SCATTER PLOT
# ==========================================

def create_scatter(
    df,
    x_col,
    y_col,
    title,
    color=None,
    size=None
):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        size=size,
        title=title
    )

    return fig


# ==========================================
# BOX PLOT
# ==========================================

def create_box_plot(
    df,
    x_col,
    y_col,
    title,
    color=None
):

    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=color,
        title=title
    )

    return fig


# ==========================================
# LINE CHART
# ==========================================

def create_line_chart(
    df,
    x_col,
    y_col,
    title,
    color=None
):

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color,
        title=title
    )

    return fig


# ==========================================
# CORRELATION HEATMAP
# ==========================================

def create_correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    return fig


# ==========================================
# GAUGE CHART
# ==========================================

def create_gauge_chart(
    value,
    title,
    min_value=0,
    max_value=100
):

    fig = go.Figure(

        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {
                    "range": [
                        min_value,
                        max_value
                    ]
                }
            }
        )

    )

    return fig


# ==========================================
# KPI CARD DATA
# ==========================================

def get_kpi_metrics(df):

    metrics = {

        "total_records":
            len(df),

        "missing_values":
            int(
                df.isnull()
                .sum()
                .sum()
            ),

        "duplicates":
            int(
                df.duplicated()
                .sum()
            )
    }

    return metrics


# ==========================================
# INCIDENT DISTRIBUTION
# ==========================================

def incident_distribution(df):

    incident_df = (
        df["incident_type"]
        .value_counts()
        .reset_index()
    )

    incident_df.columns = [
        "Incident Type",
        "Count"
    ]

    return incident_df


# ==========================================
# SEVERITY DISTRIBUTION
# ==========================================

def severity_distribution(df):

    severity_df = (
        df["emergency_severity"]
        .value_counts()
        .reset_index()
    )

    severity_df.columns = [
        "Severity",
        "Count"
    ]

    return severity_df


# ==========================================
# RISK SCORE CHART
# ==========================================

def risk_distribution_chart(df):

    if "risk_score" not in df.columns:

        return None

    fig = px.histogram(
        df,
        x="risk_score",
        nbins=40,
        title="Risk Score Distribution"
    )

    return fig


# ==========================================
# COVERAGE ANALYSIS
# ==========================================

def coverage_chart(df):

    if "coverage_percentage" not in df.columns:

        return None

    fig = px.histogram(
        df,
        x="coverage_percentage",
        nbins=30,
        title="Coverage Distribution"
    )

    return fig


# ==========================================
# RESPONSE TIME ANALYSIS
# ==========================================

def response_time_chart(df):

    if "response_time_sec" not in df.columns:

        return None

    fig = px.histogram(
        df,
        x="response_time_sec",
        nbins=30,
        title="Response Time Distribution"
    )

    return fig


# ==========================================
# TOP RISK ALERTS
# ==========================================

def top_risk_alerts(
    df,
    top_n=10
):

    if "risk_score" not in df.columns:

        return pd.DataFrame()

    return (
        df.sort_values(
            "risk_score",
            ascending=False
        )
        .head(top_n)
    )
