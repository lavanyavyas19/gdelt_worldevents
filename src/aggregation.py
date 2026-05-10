import pandas as pd


def aggregate_by(
    df: pd.DataFrame,
    granularity: str = "month_label",
) -> pd.DataFrame:
    """
    Aggregate events by country + time period.

    Parameters
    ----------
    granularity : one of 'month_label', 'week_label', or 'day'

    Returns
    -------
    DataFrame with period, country, total_events, conflict/cooperation counts,
    avg_tone, avg_goldstein, ratios.
    """
    grouped = df.groupby([granularity, "country"]).agg(
        total_events=("GLOBALEVENTID", "count"),
        conflict_events=("EventType", lambda x: (x == "Conflict").sum()),
        cooperation_events=("EventType", lambda x: (x == "Cooperation").sum()),
        avg_tone=("AvgTone", "mean"),
        avg_goldstein=("GoldsteinScale", "mean"),
        total_mentions=("NumMentions", "sum"),
    ).reset_index()

    grouped = grouped.rename(columns={granularity: "period"})

    grouped["conflict_ratio"] = (
        grouped["conflict_events"] / grouped["total_events"].clip(lower=1)
    ).round(4)
    grouped["cooperation_ratio"] = (
        grouped["cooperation_events"] / grouped["total_events"].clip(lower=1)
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
