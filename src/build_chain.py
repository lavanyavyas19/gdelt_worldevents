"""
build_chain.py
--------------
Simple event-chain retrieval system.
Given a selected event, find related events within a time window
and rank them by actor overlap, country, event type, and temporal proximity.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


# ── Scoring weights ─────────────────────────────────────────────────────────
SCORE_SAME_ACTOR = 3
SCORE_SAME_COUNTRY = 2
SCORE_SAME_ROOT_CODE = 2
SCORE_TIME_CLOSE = 1  # awarded per day of proximity (max = window size)


def find_chain(
    df: pd.DataFrame,
    event_id: int,
    window_days: int = 5,
    top_n: int = 5,
) -> dict:
    """
    Build an event chain around a selected event.

    Parameters
    ----------
    df : DataFrame
        Full cleaned + engineered event dataframe.
    event_id : int
        GLOBALEVENTID of the selected (anchor) event.
    window_days : int
        Look ±this many days around the anchor event.
    top_n : int
        Number of previous / next events to return.

    Returns
    -------
    dict with keys: 'previous', 'selected', 'next'
    """
    anchor = df[df["GLOBALEVENTID"] == event_id]
    if anchor.empty:
        return {"previous": [], "selected": None, "next": []}

    anchor_row = anchor.iloc[0]
    anchor_date = anchor_row["SQLDATE"]

    # Time window
    start = anchor_date - timedelta(days=window_days)
    end = anchor_date + timedelta(days=window_days)

    candidates = df[
        (df["SQLDATE"] >= start)
        & (df["SQLDATE"] <= end)
        & (df["GLOBALEVENTID"] != event_id)
    ].copy()

    if candidates.empty:
        return {
            "previous": [],
            "selected": anchor_row.to_dict(),
            "next": [],
        }

    # ── Score each candidate ────────────────────────────────────────────
    scores = pd.Series(0.0, index=candidates.index)

    # Same actor (either direction)
    actor_match = (
        (candidates["Actor1Name"] == anchor_row["Actor1Name"])
        | (candidates["Actor2Name"] == anchor_row["Actor2Name"])
        | (candidates["Actor1Name"] == anchor_row["Actor2Name"])
        | (candidates["Actor2Name"] == anchor_row["Actor1Name"])
    )
    scores += actor_match.astype(int) * SCORE_SAME_ACTOR

    # Same country
    scores += (
        candidates["ActionGeo_CountryCode"] == anchor_row["ActionGeo_CountryCode"]
    ).astype(int) * SCORE_SAME_COUNTRY

    # Same EventRootCode
    scores += (
        candidates["EventRootCode"] == anchor_row["EventRootCode"]
    ).astype(int) * SCORE_SAME_ROOT_CODE

    # Temporal proximity bonus (closer = higher)
    day_diff = (candidates["SQLDATE"] - anchor_date).dt.days.abs()
    time_bonus = ((window_days - day_diff) / window_days).clip(0, 1) * SCORE_TIME_CLOSE
    scores += time_bonus

    candidates["chain_score"] = scores.round(3)

    # Split into before / after
    before = (
        candidates[candidates["SQLDATE"] < anchor_date]
        .nlargest(top_n, "chain_score")
    )
    after = (
        candidates[candidates["SQLDATE"] > anchor_date]
        .nlargest(top_n, "chain_score")
    )

    cols = [
        "GLOBALEVENTID", "SQLDATE", "Actor1Name", "Actor2Name",
        "country", "EventRootCode", "QuadLabel", "EventType",
        "AvgTone", "chain_score", "SOURCEURL",
    ]
    sel_cols = [c for c in cols if c in anchor.columns]

    return {
        "previous": before[sel_cols].to_dict("records"),
        "selected": anchor_row[sel_cols].to_dict(),
        "next": after[sel_cols].to_dict("records"),
    }
