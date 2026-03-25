"""
evaluation.py
-------------
Sanity checks and quality metrics for the GDELT Event Intelligence Dashboard.

This is NOT classification accuracy. We evaluate:
  • Data quality: duplicates removed, invalid dates, date coverage
  • Burst detection: burst count, ratio, z-score distribution
  • Chain retrieval: sample chain quality metrics
  • Keyword quality: before/after cleaning examples
  • Response time: chain build latency
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .chains import find_chain


def evaluate_data_quality(df: pd.DataFrame, report: dict) -> dict:
    """
    Data quality metrics from preprocessing.
    """
    return {
        "rows_raw": report.get("rows_raw", 0),
        "rows_final": report.get("rows_final", 0),
        "invalid_dates_dropped": report.get("invalid_dates_dropped", 0),
        "duplicates_removed": report.get("duplicates_removed", 0),
        "rows_outside_countries": report.get("rows_outside_countries", 0),
        "rows_outside_window": report.get("rows_outside_window", 0),
        "date_range": f"{report.get('date_min', '?')} → {report.get('date_max', '?')}",
        "countries": report.get("countries", {}),
        "unique_days": int(df["day"].nunique()) if "day" in df.columns else 0,
        "missing_actor_pct": round(
            float(df["has_missing_actor"].mean() * 100), 1
        ) if "has_missing_actor" in df.columns else 0,
    }


def evaluate_burst_sanity(burst_df: pd.DataFrame) -> dict:
    """
    Sanity checks on burst detection output.
    """
    total_rows = len(burst_df)
    burst_rows = int(burst_df["is_burst"].sum())
    total_days = burst_df["day"].nunique()

    return {
        "total_country_days": total_rows,
        "calendar_days": int(total_days),
        "burst_country_days": burst_rows,
        "burst_ratio": round(burst_rows / max(total_rows, 1), 4),
        "mean_z_on_burst": round(
            float(burst_df[burst_df["is_burst"]]["z_score"].mean()), 3
        ) if burst_rows > 0 else 0,
        "max_z_score": round(float(burst_df["z_score"].max()), 3),
        "bursts_per_country": (
            burst_df[burst_df["is_burst"]]
            .groupby("country")["is_burst"]
            .sum()
            .to_dict()
        ) if burst_rows > 0 else {},
    }


def evaluate_chain_sample(
    df: pd.DataFrame,
    sample_ids: list,
    window_days: int = 7,
    top_n: int = 5,
) -> list:
    """
    For a sample of events, build chains and compute quality metrics.
    """
    results = []
    for eid in sample_ids:
        chain = find_chain(df, int(eid), window_days, top_n)
        if chain["selected"] is None:
            continue

        all_linked = chain["previous"] + chain["next"]
        if not all_linked:
            results.append({
                "event_id": int(eid),
                "chain_size": 0,
                "actor_overlap": 0,
                "country_overlap": 0,
                "avg_chain_score": 0,
            })
            continue

        anchor = chain["selected"]
        anchor_actors = {
            str(anchor.get("actor1_clean", "")),
            str(anchor.get("actor2_clean", "")),
        } - {"Unknown", ""}
        anchor_country = anchor.get("country", "")

        actor_hits = sum(
            1 for e in all_linked
            if str(e.get("actor1_clean", "")) in anchor_actors
            or str(e.get("actor2_clean", "")) in anchor_actors
        )
        country_hits = sum(
            1 for e in all_linked if e.get("country") == anchor_country
        )
        avg_score = sum(e.get("chain_score", 0) for e in all_linked) / len(all_linked)

        results.append({
            "event_id": int(eid),
            "chain_size": len(all_linked),
            "actor_overlap": round(actor_hits / len(all_linked), 3),
            "country_overlap": round(country_hits / len(all_linked), 3),
            "avg_chain_score": round(avg_score, 3),
        })
    return results


def evaluate_response_time(df: pd.DataFrame, event_id: int, n_runs: int = 3) -> float:
    """Average chain-build time in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        find_chain(df, int(event_id))
        times.append(time.time() - t0)
    return round(sum(times) / len(times), 4)


def build_evaluation_summary(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    cleaning_report: dict,
) -> dict:
    """Build a complete evaluation summary."""
    summary = {
        "data_quality": evaluate_data_quality(df, cleaning_report),
        "burst_sanity": evaluate_burst_sanity(burst_df),
    }

    # Chain sample evaluation
    sample_ids = (
        df["GLOBALEVENTID"]
        .dropna()
        .sample(min(15, len(df)), random_state=42)
        .tolist()
    )
    if sample_ids:
        summary["chain_evaluation"] = evaluate_chain_sample(df, sample_ids)
        summary["avg_response_time_sec"] = evaluate_response_time(
            df, sample_ids[0]
        )

    return summary


def save_evaluation(results: dict, path: str) -> None:
    """Save evaluation outputs as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  Evaluation saved → {path}")
