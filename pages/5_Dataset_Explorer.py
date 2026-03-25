"""
Page 5 — Dataset Explorer
Filterable event table with date, country, event type, tone, and source links.
"""

import streamlit as st
import pandas as pd
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.storage import load_df
PROCESSED = os.path.join(ROOT, "data", "processed")


@st.cache_data
def load_events():
    return load_df(os.path.join(PROCESSED, "events"))


st.header("Dataset Explorer")

try:
    df = load_events()
except FileNotFoundError:
    st.error("Processed data not found."); st.stop()

# ── Sidebar filters ─────────────────────────────────────────────────────────
countries = st.sidebar.multiselect(
    "Countries", df["country"].unique().tolist(),
    default=df["country"].unique().tolist(), key="exp_countries",
)
event_types = st.sidebar.multiselect(
    "Event Type", ["Conflict", "Cooperation"],
    default=["Conflict", "Cooperation"], key="exp_etype",
)

min_date = df["SQLDATE"].min().date()
max_date = df["SQLDATE"].max().date()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date, key="exp_dates",
)

tone_range = st.sidebar.slider(
    "Tone Range",
    float(df["AvgTone"].min()), float(df["AvgTone"].max()),
    (float(df["AvgTone"].min()), float(df["AvgTone"].max())),
    key="exp_tone",
)

# ── Apply filters ───────────────────────────────────────────────────────────
mask = (
    df["country"].isin(countries)
    & df["EventType"].isin(event_types)
    & (df["AvgTone"] >= tone_range[0])
    & (df["AvgTone"] <= tone_range[1])
)
if len(date_range) == 2:
    mask &= (df["SQLDATE"].dt.date >= date_range[0]) & (df["SQLDATE"].dt.date <= date_range[1])

filtered = df[mask]

st.write(f"**{len(filtered):,}** events match your filters.")

# ── Display columns ─────────────────────────────────────────────────────────
display_cols = [
    "SQLDATE", "country", "Actor1Name", "Actor2Name",
    "EventType", "QuadLabel", "AvgTone", "NumMentions", "SOURCEURL",
]
existing = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[existing].head(500),
    use_container_width=True,
    column_config={
        "SOURCEURL": st.column_config.LinkColumn("Source"),
        "AvgTone": st.column_config.NumberColumn(format="%.2f"),
    },
)

if len(filtered) > 500:
    st.caption("Showing first 500 rows. Use filters to narrow results.")
