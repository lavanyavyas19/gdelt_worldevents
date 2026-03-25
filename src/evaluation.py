"""
evaluation.py
-------------
Retrieval-quality and system evaluation utilities.
NOT classification accuracy — we evaluate:
  • chain relevance
  • burst detection sanity
  • top-actor consistency
  • response time
"""

import time
import json
import pandas as pd
from pathlib import Path
from .build_chain import find_chain


def evaluate_chain_sample(
    df: pd.DataFrame,
    sample_ids: list,
    window_days: int = 5,
    top_n: int = 5,
) -> list:
    """
    For each sample event, build a chain and compute simple metrics:
      - chain_size: how many related events found
      - actor_overlap: fraction of chain events sharing ≥1 actor
      - country_overlap: fraction sharing same country
      - avg_chain_score: mean relevance score
    """
    results = []
    for eid in sample_ids:
        chain = find_chain(df, eid, window_days, top_n)
        if chain["selected"] is None:
            continue

        all_linked = chain["previous"] + chain["next"]
        if not all_linked:
            results.append({
                "event_id": eid,
                "chain_size": 0,
                "actor_overlap": 0,
                "country_overlap": 0,
                "avg_chain_score": 0,
            })
            continue

        anchor = chain["selected"]
        actor_set = {anchor.get("Actor1Name"), anchor.get("Actor2Name")}
        anchor_country = anchor.get("country")

        actor_hits = sum(
            1 for e in all_linked
            if e.get("Actor1Name") in actor_set or e.get("Actor2Name") in actor_set
        )
        country_hits = sum(
            1 for e in all_linked if e.get("country") == anchor_country
        )
        avg_score = sum(e.get("chain_score", 0) for e in all_linked) / len(all_linked)

        results.append({
            "event_id": eid,
            "chain_size": len(all_linked),
            "actor_overlap": round(actor_hits / len(all_linked), 3),
            "country_overlap": round(country_hits / len(all_linked), 3),
            "avg_chain_score": round(avg_score, 3),
        })
    return results


def evaluate_burst_sanity(burst_df: pd.DataFrame) -> dict:
    """
    Basic sanity checks on burst detection output.
    """
    total_days = burst_df[["day", "country"]].drop_duplicates().shape[0]
    burst_days = burst_df[burst_df["is_burst"]].shape[0]
    return {
        "total_country_days": int(total_days),
        "burst_country_days": int(burst_days),
        "burst_ratio": round(burst_days / max(total_days, 1), 4),
        "mean_z_on_burst": round(
            float(burst_df[burst_df["is_burst"]]["z_score"].mean()), 3
        ) if burst_days > 0 else 0,
    }


def evaluate_response_time(df: pd.DataFrame, event_id: int, n_runs: int = 3) -> float:
    """Average chain-build time in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        find_chain(df, event_id)
        times.append(time.time() - t0)
    return round(sum(times) / len(times), 4)


def save_evaluation(results: dict, path: str) -> None:
    """Save evaluation outputs as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serialisable values
    def _convert(obj):
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  Evaluation saved → {path}")
