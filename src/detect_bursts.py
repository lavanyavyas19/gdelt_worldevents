"""
detect_bursts.py
----------------
Statistical burst detection using z-score on daily event counts.
A "burst" is a day where the event count deviates significantly
from the rolling average — signaling an unusual spike in activity.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Default burst detection parameters
DEFAULT_RULES = {
    "rolling_window": 7,
    "z_threshold": 2.0,
    "min_events": 10,
}


def save_burst_rules(rules: dict, path: str) -> None:
    """Persist burst detection rules as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"  Burst rules saved → {path}")


def detect_bursts(
    df: pd.DataFrame,
    rules: dict = None,
) -> pd.DataFrame:
    """
    Detect burst days per country.

    Algorithm
    ---------
    1. Count events per country per day
    2. Compute rolling mean & std (window = rules['rolling_window'])
    3. Calculate z-score = (count - rolling_mean) / rolling_std
    4. Flag burst if z_score > rules['z_threshold']

    Returns
    -------
    DataFrame with columns:
        day, country, event_count, rolling_mean, rolling_std,
        z_score, is_burst
    """
    if rules is None:
        rules = DEFAULT_RULES

    window = rules["rolling_window"]
    threshold = rules["z_threshold"]
    min_ev = rules.get("min_events", 10)

    # Daily counts per country
    daily = (
        df.groupby(["day", "country"])
        .agg(event_count=("GLOBALEVENTID", "count"))
        .reset_index()
        .sort_values(["country", "day"])
    )

    results = []
    for country, grp in daily.groupby("country"):
        grp = grp.copy().sort_values("day")
        grp["rolling_mean"] = grp["event_count"].rolling(window, min_periods=1).mean()
        grp["rolling_std"] = grp["event_count"].rolling(window, min_periods=1).std().fillna(1)
        grp["z_score"] = (
            (grp["event_count"] - grp["rolling_mean"]) / grp["rolling_std"]
        ).round(3)
        grp["is_burst"] = (grp["z_score"] > threshold) & (grp["event_count"] >= min_ev)
        results.append(grp)

    burst_df = pd.concat(results, ignore_index=True)
    n_bursts = burst_df["is_burst"].sum()
    print(f"  Burst detection complete: {n_bursts} burst-days found")
    return burst_df
