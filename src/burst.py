import json
import pandas as pd
import numpy as np
from pathlib import Path

from .config import BURST_ROLLING_WINDOW, BURST_Z_THRESHOLD, BURST_MIN_EVENTS


def detect_bursts(
    df: pd.DataFrame,
    rolling_window: int = BURST_ROLLING_WINDOW,
    z_threshold: float = BURST_Z_THRESHOLD,
    min_events: int = BURST_MIN_EVENTS,
) -> pd.DataFrame:
    """
    Detect burst days per country with complete date coverage.

    Algorithm
    ---------
    1. Determine the full date range from the dataset (min → max event_date)
    2. For each country, create a row for every day in that range
    3. Count events per country per day (fill missing = 0)
    4. Compute rolling mean & std over the filled series
    5. Compute z-score = (count - rolling_mean) / rolling_std
    6. Flag burst if z_score > threshold AND count >= min_events

    Returns
    -------
    DataFrame: day, country, event_count, rolling_mean, rolling_std, z_score, is_burst
    """
    # Ensure day column is datetime
    if "day" not in df.columns:
        df = df.copy()
        df["day"] = pd.to_datetime(df["event_date"]).dt.normalize()

    # Daily counts per country
    daily = (
        df.groupby(["day", "country"])
        .agg(event_count=("GLOBALEVENTID", "count"))
        .reset_index()
    )

    # Complete date range across all data
    date_min = df["day"].min()
    date_max = df["day"].max()
    all_days = pd.date_range(start=date_min, end=date_max, freq="D")
    countries = df["country"].dropna().unique()

    # Build complete grid: every day × every country
    grid = pd.MultiIndex.from_product(
        [all_days, countries], names=["day", "country"]
    ).to_frame(index=False)

    # Merge actual counts onto the complete grid
    merged = grid.merge(daily, on=["day", "country"], how="left")
    merged["event_count"] = merged["event_count"].fillna(0).astype(int)

    # Per-country rolling statistics
    results = []
    for country, grp in merged.groupby("country"):
        grp = grp.sort_values("day").copy()

        # LAGGED rolling stats: shift(1) so the current day is NOT included
        # in its own baseline. This prevents spikes from inflating their own mean.
        grp["rolling_mean"] = (
            grp["event_count"]
            .rolling(rolling_window, min_periods=1)
            .mean()
            .shift(1)
            .fillna(grp["event_count"].expanding(min_periods=1).mean().shift(1))
            .fillna(0)
            .round(2)
        )
        grp["rolling_std"] = (
            grp["event_count"]
            .rolling(rolling_window, min_periods=1)
            .std()
            .shift(1)
            .fillna(grp["event_count"].expanding(min_periods=1).std().shift(1))
            .fillna(0)
            .round(2)
        )

        # Z-score with safe divide
        grp["z_score"] = np.where(
            grp["rolling_std"] > 0,
            ((grp["event_count"] - grp["rolling_mean"]) / grp["rolling_std"]).round(3),
            0.0,
        )

        # Burst flag
        grp["is_burst"] = (
            (grp["z_score"] > z_threshold) & (grp["event_count"] >= min_events)
        )
        results.append(grp)

    burst_df = pd.concat(results, ignore_index=True)
    n_bursts = burst_df["is_burst"].sum()
    print(f"  Burst detection: {n_bursts} burst-days across "
          f"{len(all_days)} calendar days × {len(countries)} countries")
    return burst_df


def get_burst_summary(burst_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table of burst days with key metrics."""
    bursts = burst_df[burst_df["is_burst"]].copy()
    if bursts.empty:
        return pd.DataFrame(columns=["day", "country", "event_count", "z_score"])
    return (
        bursts[["day", "country", "event_count", "rolling_mean", "z_score"]]
        .sort_values("z_score", ascending=False)
        .reset_index(drop=True)
    )


def burst_events_detail(
    df: pd.DataFrame, burst_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return the actual events that occurred on burst days,
    useful for drilling into what happened during a spike.
    """
    burst_days = burst_df[burst_df["is_burst"]][["day", "country"]].copy()
    if burst_days.empty:
        return pd.DataFrame()
    # Merge on day + country
    return df.merge(burst_days, on=["day", "country"], how="inner")


def save_burst_rules(rules: dict, path: str) -> None:
    """Persist burst detection parameters as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"  Burst rules saved → {path}")
