"""
utils.py
--------
Shared utility functions for the Streamlit dashboard pages.
Centralises data loading, filtering, UI helpers, and display formatting.
"""

import os
import pickle
import streamlit as st
import pandas as pd

from .storage import load_df
from .config import (
    PROCESSED_DIR, MODELS_DIR,
    TARGET_COUNTRY_NAMES, COLOR_MAP_COUNTRY, COLOR_MAP_EVENT,
    DATA_WINDOW_LABEL,
)


# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading events…")
def load_events() -> pd.DataFrame:
    """Load the main processed events DataFrame."""
    df = load_df(os.path.join(PROCESSED_DIR, "events"))
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"])
    return df


@st.cache_data(show_spinner="Loading burst data…")
def load_bursts() -> pd.DataFrame:
    """Load pre-computed burst detection results."""
    bdf = load_df(os.path.join(PROCESSED_DIR, "bursts"))
    if "day" in bdf.columns:
        bdf["day"] = pd.to_datetime(bdf["day"])
    return bdf


@st.cache_resource(show_spinner="Loading keyword model…")
def load_vectorizer():
    """Load the fitted TF-IDF vectorizer."""
    path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Tone helpers ──────────────────────────────────────────────────────────────

def tone_label(val) -> str:
    """Convert numeric AvgTone to a human-readable label."""
    if pd.isna(val):
        return "Unknown"
    v = float(val)
    if v < -3:
        return "Very Negative"
    if v < -0.3:
        return "Negative"
    if v <= 0.3:
        return "Neutral"
    if v <= 3:
        return "Positive"
    return "Very Positive"


def tone_with_value(val) -> str:
    """Label with number, e.g. 'Negative (-1.4)'."""
    if pd.isna(val):
        return "Unknown"
    return f"{tone_label(val)} ({float(val):.1f})"


def match_strength(score: float, max_score: float = 15.0) -> str:
    """Convert a raw chain score into a human-friendly strength label."""
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.65:
        return "Strong"
    if ratio >= 0.4:
        return "Moderate"
    return "Weak"


# ── Column display names (raw → human) ───────────────────────────────────────

COLUMN_LABELS = {
    "event_date": "Date",
    "country": "Country",
    "actor1_clean": "Actor 1",
    "actor2_clean": "Actor 2",
    "EventType": "Type",
    "QuadLabel": "Event Class",
    "EventRootLabel": "Event Family",
    "AvgTone": "Tone",
    "NumMentions": "Mentions",
    "ActionGeo_FullName": "Location",
    "SOURCEURL": "Source",
    "event_count": "Events",
    "rolling_mean": "Typical Level",
    "rolling_std": "Variability",
    "z_score": "Burst Strength",
    "is_burst": "Spike?",
    "burst_days": "Spike Days",
    "avg_z": "Avg Burst Strength",
    "max_events": "Peak Events",
    "chain_score": "Relevance",
    "total_events": "Events",
    "conflict_events": "Conflict",
    "cooperation_events": "Cooperation",
    "avg_tone": "Avg Tone",
    "avg_goldstein": "Avg Goldstein",
    "conflict_ratio": "Conflict Share",
    "cooperation_ratio": "Cooperation Share",
    "total_mentions": "Total Mentions",
    "events": "Events",
    "source_domain": "Source",
}


def friendly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns using human-friendly labels for display."""
    return df.rename(columns={k: v for k, v in COLUMN_LABELS.items() if k in df.columns})


# ── Score reason formatting ───────────────────────────────────────────────────

REASON_LABELS = {
    "country": "Same country",
    "actor": "Shared actor",
    "event_family": "Similar event type",
    "quad_class": "Same conflict/cooperation class",
    "location": "Same location",
    "tone": "Similar tone",
}


def format_reasons(raw_reasons: str) -> list[str]:
    """Convert 'country, actor, tone, ' into ['Same country', 'Shared actor', 'Similar tone']."""
    if not raw_reasons:
        return []
    keys = [r.strip() for r in raw_reasons.strip(", ").split(",") if r.strip()]
    return [REASON_LABELS.get(k, k) for k in keys]


# ── Sidebar helpers ───────────────────────────────────────────────────────────

def sidebar_country_filter(df: pd.DataFrame, key: str = "countries") -> list:
    available = sorted(df["country"].dropna().unique().tolist())
    return st.sidebar.multiselect(
        "Countries",
        available,
        default=available,
        key=key,
    )


def sidebar_event_type_filter(key: str = "event_type") -> list:
    return st.sidebar.multiselect(
        "Event Type",
        ["Conflict", "Cooperation"],
        default=["Conflict", "Cooperation"],
        key=key,
    )


def show_data_window():
    st.sidebar.markdown(f"**{DATA_WINDOW_LABEL}**")
    st.sidebar.divider()


def apply_filters(df: pd.DataFrame, countries: list, event_types: list = None) -> pd.DataFrame:
    mask = df["country"].isin(countries)
    if event_types:
        mask &= df["EventType"].isin(event_types)
    return df[mask]


# ── UI helpers ────────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = ""):
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def empty_state(message: str = "No data matches your current filters."):
    st.info(message)


def metric_row(metrics: list[tuple]):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def data_not_found():
    st.error(
        "Processed data not found. Please run the data pipeline first:\n\n"
        "```bash\npython -m src.prepare_data\n```"
    )
    st.stop()
