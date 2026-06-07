import pandas as pd
import numpy as np

# ==========================================
# DATASET OVERVIEW INSIGHTS
# ==========================================

def get_overview_insights(df):

    insights = []

    insights.append(
        f"Total alerts analyzed: {len(df)}"
    )

    if "incident_type" in df.columns:

        most_common = (
            df["incident_type"]
            .mode()[0]
        )

        insights.append(
            f"Most common incident type: {most_common}"
        )

    if "emergency_severity" in df.columns:

        severity = (
            df["emergency_severity"]
            .mode()[0]
        )

        insights.append(
            f"Most frequent severity level: {severity}"
        )

    return insights


# ==========================================
# SENSOR INSIGHTS
# ==========================================

def get_sensor_insights(df):

    insights = []

    if "temperature_celsius" in df.columns:

        insights.append(
            f"Maximum temperature recorded: "
            f"{round(df['temperature_celsius'].max(),2)} °C"
        )

    if "smoke_density_ppm" in df.columns:

        insights.append(
            f"Maximum smoke density: "
            f"{round(df['smoke_density_ppm'].max(),2)}"
        )

    if "co2_level_ppm" in df.columns:

        insights.append(
            f"Maximum CO₂ level: "
            f"{round(df['co2_level_ppm'].max(),2)} ppm"
        )

    return insights


# ==========================================
# RESPONSE INSIGHTS
# ==========================================

def get_response_insights(df):

    insights = []

    if "response_time_sec" in df.columns:

        avg_response = round(
            df["response_time_sec"].mean(),
            2
        )

        insights.append(
            f"Average response time: "
            f"{avg_response} seconds"
        )

    if "coverage_percentage" in df.columns:

        avg_coverage = round(
            df["coverage_percentage"].mean(),
            2
        )

        insights.append(
            f"Average alert coverage: "
            f"{avg_coverage}%"
        )

    return insights


# ==========================================
# RELIABILITY INSIGHTS
# ==========================================

def get_reliability_insights(df):

    insights = []

    if "alert_reliability_score" in df.columns:

        avg_rel = round(
            df["alert_reliability_score"]
            .mean() * 100,
            2
        )

        insights.append(
            f"System reliability: "
            f"{avg_rel}%"
        )

    return insights


# ==========================================
# PROPAGATION INSIGHTS
# ==========================================

def get_propagation_insights(df):

    insights = []

    if "propagation_delay_sec" in df.columns:

        avg_delay = round(
            df["propagation_delay_sec"]
            .mean(),
            2
        )

        insights.append(
            f"Average propagation delay: "
            f"{avg_delay} sec"
        )

    if "social_alerts_forwarded" in df.columns:

        forwarded = round(
            df["social_alerts_forwarded"]
            .mean(),
            2
        )

        insights.append(
            f"Average forwarded alerts: "
            f"{forwarded}"
        )

    if "node_influence_score" in df.columns:

        influence = round(
            df["node_influence_score"]
            .mean(),
            2
        )

        insights.append(
            f"Average node influence score: "
            f"{influence}"
        )

    return insights


# ==========================================
# RISK INSIGHTS
# ==========================================

def get_risk_insights(df):

    insights = []

    if "risk_score" in df.columns:

        avg_risk = round(
            df["risk_score"].mean(),
            2
        )

        max_risk = round(
            df["risk_score"].max(),
            2
        )

        insights.append(
            f"Average risk score: {avg_risk}"
        )

        insights.append(
            f"Maximum risk score: {max_risk}"
        )

    return insights


# ==========================================
# CRITICAL ALERTS
# ==========================================

def get_critical_alert_insights(df):

    insights = []

    if "emergency_severity" in df.columns:

        critical_count = len(

            df[
                df["emergency_severity"]
                .astype(str)
                .str.lower()
                == "critical"
            ]

        )

        critical_percent = round(

            (
                critical_count
                / len(df)
            ) * 100,

            2

        )

        insights.append(
            f"Critical alerts: "
            f"{critical_count}"
        )

        insights.append(
            f"Critical alert percentage: "
            f"{critical_percent}%"
        )

    return insights


# ==========================================
# TOP RISK ALERTS
# ==========================================

def get_top_risk_alerts(
    df,
    top_n=10
):

    if "risk_score" not in df.columns:

        return pd.DataFrame()

    return (
        df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(top_n)
    )


# ==========================================
# EXECUTIVE SUMMARY
# ==========================================

def generate_executive_summary(df):

    summary = []

    summary.extend(
        get_overview_insights(df)
    )

    summary.extend(
        get_sensor_insights(df)
    )

    summary.extend(
        get_response_insights(df)
    )

    summary.extend(
        get_reliability_insights(df)
    )

    summary.extend(
        get_propagation_insights(df)
    )

    summary.extend(
        get_risk_insights(df)
    )

    summary.extend(
        get_critical_alert_insights(df)
    )

    return summary


# ==========================================
# TEXT REPORT
# ==========================================

def generate_text_report(df):

    summary = generate_executive_summary(df)

    report = (
        "EMERGENCY ALERT "
        "INTELLIGENCE REPORT\n\n"
    )

    for item in summary:

        report += f"- {item}\n"

    return report
