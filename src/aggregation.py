"""
aggregation.py
--------------
Aggregate event-level data into country × time summaries.
Supports monthly and weekly granularity.
"""

import pandas as pd


def aggregate_by(
    df: pd.DataFrame,
    time_col: str = "month",  # or "week"
) -> pd.DataFrame:
    """
    Aggregate events by country + time period.

    Returns
    -------
    DataFrame with columns:
        period, country, total_events, conflict_events, cooperation_events,
        avg_tone, avg_goldstein, conflict_ratio, cooperation_ratio
    """
    grouped = df.groupby([time_col, "country"]).agg(
        total_events=("GLOBALEVENTID", "count"),
        conflict_events=("EventType", lambda x: (x == "Conflict").sum()),
        cooperation_events=("EventType", lambda x: (x == "Cooperation").sum()),
        avg_tone=("AvgTone", "mean"),
        avg_goldstein=("GoldsteinScale", "mean"),
        total_mentions=("NumMentions", "sum"),
    ).reset_index()

    grouped = grouped.rename(columns={time_col: "period"})

    # Ratios
    grouped["conflict_ratio"] = (
        grouped["conflict_events"] / grouped["total_events"]
    ).round(4)
    grouped["cooperation_ratio"] = (
        grouped["cooperation_events"] / grouped["total_events"]
    ).round(4)

    return grouped.sort_values(["period", "country"]).reset_index(drop=True)


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """High-level summary per country (for KPI cards)."""
    return df.groupby("country").agg(
        total_events=("GLOBALEVENTID", "count"),
        conflict_events=("EventType", lambda x: (x == "Conflict").sum()),
        cooperation_events=("EventType", lambda x: (x == "Cooperation").sum()),
        avg_tone=("AvgTone", "mean"),
        avg_goldstein=("GoldsteinScale", "mean"),
    ).reset_index()
