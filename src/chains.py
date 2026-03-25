"""
chains.py
---------
Scoring-based event chain retrieval.

Given an anchor event, this module finds the most related events
before and after it using a multi-factor relevance score:
  +3 same country
  +3 same actor (any direction)
  +2 same EventRootCode (event family)
  +2 same QuadLabel
  +1 same ActionGeo_FullName or similar location
  +1 similar AvgTone (within 3 points)
  +1 time proximity bonus (closer = higher, scaled 0–1)

This replaces the old build_chain.py.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from .config import (
    CHAIN_SCORE_SAME_COUNTRY,
    CHAIN_SCORE_SAME_ACTOR,
    CHAIN_SCORE_SAME_EVENT_TYPE,
    CHAIN_SCORE_SAME_QUAD,
    CHAIN_SCORE_SAME_LOCATION,
    CHAIN_SCORE_TONE_SIMILAR,
    CHAIN_SCORE_TIME_PROXIMITY,
)


def find_chain(
    df: pd.DataFrame,
    event_id: int,
    window_days: int = 7,
    top_n: int = 5,
    country_filter: str = None,
) -> dict:
    """
    Build a scored event chain around an anchor event.

    Parameters
    ----------
    df            : full cleaned + engineered event DataFrame
    event_id      : GLOBALEVENTID of the anchor event
    window_days   : look ±this many days around the anchor
    top_n         : number of previous/next events to return
    country_filter: if set, restrict candidates to this country

    Returns
    -------
    dict with keys: 'previous', 'selected', 'next', 'explanation'
    Each linked event includes a 'chain_score' and 'score_reasons' field.
    """
    anchor = df[df["GLOBALEVENTID"] == event_id]
    empty = {"previous": [], "selected": None, "next": [], "explanation": ""}

    if anchor.empty:
        return empty

    anchor_row = anchor.iloc[0]
    anchor_date = anchor_row["event_date"]

    if pd.isna(anchor_date):
        return empty

    # Time window
    start = anchor_date - timedelta(days=window_days)
    end = anchor_date + timedelta(days=window_days)

    # Candidate pool
    mask = (
        (df["event_date"] >= start)
        & (df["event_date"] <= end)
        & (df["GLOBALEVENTID"] != event_id)
    )
    if country_filter:
        mask &= df["country"] == country_filter

    candidates = df[mask].copy()

    if candidates.empty:
        return {
            "previous": [],
            "selected": _format_anchor(anchor_row),
            "next": [],
            "explanation": "No candidate events found in the time window.",
        }

    # ── Score each candidate ──────────────────────────────────────────────
    scores = pd.Series(0.0, index=candidates.index)
    reasons = pd.Series("", index=candidates.index)

    # Same country (+3)
    country_match = candidates["country"] == anchor_row.get("country")
    scores += country_match.astype(int) * CHAIN_SCORE_SAME_COUNTRY
    reasons = _add_reason(reasons, country_match, "country")

    # Same actor — any direction (+3)
    a1 = _safe_str(anchor_row.get("actor1_clean"))
    a2 = _safe_str(anchor_row.get("actor2_clean"))
    anchor_actors = {a1, a2} - {"Unknown", ""}

    if anchor_actors:
        actor_match = (
            candidates["actor1_clean"].isin(anchor_actors)
            | candidates["actor2_clean"].isin(anchor_actors)
        )
        scores += actor_match.astype(int) * CHAIN_SCORE_SAME_ACTOR
        reasons = _add_reason(reasons, actor_match, "actor")

    # Same EventRootCode (+2)
    root_match = (
        candidates["EventRootCode"].astype(str) == str(anchor_row.get("EventRootCode", ""))
    )
    scores += root_match.astype(int) * CHAIN_SCORE_SAME_EVENT_TYPE
    reasons = _add_reason(reasons, root_match, "event_family")

    # Same QuadLabel (+2)
    quad_match = candidates["QuadLabel"] == anchor_row.get("QuadLabel")
    scores += quad_match.astype(int) * CHAIN_SCORE_SAME_QUAD
    reasons = _add_reason(reasons, quad_match, "quad_class")

    # Same location string (+1)
    anchor_loc = _safe_str(anchor_row.get("ActionGeo_FullName"))
    if anchor_loc:
        loc_match = candidates["ActionGeo_FullName"].fillna("") == anchor_loc
        scores += loc_match.astype(int) * CHAIN_SCORE_SAME_LOCATION
        reasons = _add_reason(reasons, loc_match, "location")

    # Similar tone: within 3 points (+1)
    anchor_tone = anchor_row.get("AvgTone", 0)
    if pd.notna(anchor_tone):
        tone_diff = (candidates["AvgTone"] - anchor_tone).abs()
        tone_match = tone_diff <= 3.0
        scores += tone_match.astype(int) * CHAIN_SCORE_TONE_SIMILAR
        reasons = _add_reason(reasons, tone_match, "tone")

    # Time proximity bonus: closer → higher (0–1 scaled)
    day_diff = (candidates["event_date"] - anchor_date).dt.total_seconds().abs() / 86400
    time_bonus = ((window_days - day_diff) / max(window_days, 1)).clip(0, 1)
    scores += (time_bonus * CHAIN_SCORE_TIME_PROXIMITY).round(3)

    candidates["chain_score"] = scores.round(3)
    candidates["score_reasons"] = reasons

    # Split before / after anchor date
    before = (
        candidates[candidates["event_date"] < anchor_date]
        .nlargest(top_n, "chain_score")
    )
    after = (
        candidates[candidates["event_date"] > anchor_date]
        .nlargest(top_n, "chain_score")
    )
    # Also include same-day events in both (if any)
    same_day = candidates[candidates["event_date"] == anchor_date]
    if not same_day.empty and len(before) < top_n:
        extra = same_day.nlargest(top_n - len(before), "chain_score")
        before = pd.concat([before, extra]).nlargest(top_n, "chain_score")

    cols = _result_columns(df)

    explanation = (
        "These events are linked because they share one or more of: "
        "country, actor overlap, event family, conflict/cooperation class, "
        "location, tone similarity, and temporal proximity."
    )

    return {
        "previous": before[cols].to_dict("records"),
        "selected": _format_anchor(anchor_row),
        "next": after[cols].to_dict("records"),
        "explanation": explanation,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _add_reason(reasons: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    """Append a reason label where the mask is True."""
    return reasons.where(~mask, reasons + label + ", ")


def _format_anchor(row) -> dict:
    """Format anchor event for display."""
    keys = [
        "GLOBALEVENTID", "event_date", "actor1_clean", "actor2_clean",
        "country", "EventRootCode", "EventRootLabel", "QuadLabel", "EventType",
        "AvgTone", "ActionGeo_FullName", "SOURCEURL", "event_label",
    ]
    result = {}
    for k in keys:
        val = row.get(k) if hasattr(row, "get") else row[k] if k in row.index else None
        if pd.isna(val):
            val = ""
        result[k] = val
    return result


def _result_columns(df: pd.DataFrame) -> list:
    """Columns to include in chain results."""
    desired = [
        "GLOBALEVENTID", "event_date", "actor1_clean", "actor2_clean",
        "country", "EventRootCode", "EventRootLabel", "QuadLabel", "EventType",
        "AvgTone", "ActionGeo_FullName", "SOURCEURL", "chain_score",
        "score_reasons", "event_label",
    ]
    return [c for c in desired if c in df.columns or c in ("chain_score", "score_reasons")]
