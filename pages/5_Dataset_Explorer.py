"""
Page 5 — Dataset Explorer
One question: What does the raw data look like?
"""

import streamlit as st
import pandas as pd
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, sidebar_country_filter, sidebar_event_type_filter,
    show_data_window, data_not_found, empty_state, friendly_columns,
    COLUMN_LABELS,
)
from src.config import EXPLORER_ROW_LIMIT

try:
    df = load_events()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(df, key="exp_countries")
event_types = sidebar_event_type_filter(key="exp_etype")

date_min = df["event_date"].min().date()
date_max = df["event_date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max),
    min_value=date_min, max_value=date_max, key="exp_dates",
)

tone_lo, tone_hi = float(df["AvgTone"].min()), float(df["AvgTone"].max())
tone_range = st.sidebar.slider(
    "Tone range", tone_lo, tone_hi, (tone_lo, tone_hi), key="exp_tone",
)

# Human-friendly column picker
raw_cols = [
    "event_date", "country", "actor1_clean", "actor2_clean",
    "EventType", "QuadLabel", "EventRootLabel", "AvgTone",
    "NumMentions", "ActionGeo_FullName", "SOURCEURL",
]
available = [c for c in raw_cols if c in df.columns]
# Show friendly names in the picker
friendly_options = {COLUMN_LABELS.get(c, c): c for c in available}
selected_friendly = st.sidebar.multiselect(
    "Columns", list(friendly_options.keys()),
    default=list(friendly_options.keys())[:8], key="exp_cols",
)
selected_cols = [friendly_options[f] for f in selected_friendly]

st.header("Browse Events")
st.caption("Filter and explore the underlying dataset.")

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df["country"].isin(countries)
    & df["EventType"].isin(event_types)
    & (df["AvgTone"] >= tone_range[0])
    & (df["AvgTone"] <= tone_range[1])
)
if len(date_range) == 2:
    mask &= (
        (df["event_date"].dt.date >= date_range[0])
        & (df["event_date"].dt.date <= date_range[1])
    )

filtered = df[mask]

st.markdown(f"**{len(filtered):,}** events match your filters")

if filtered.empty:
    empty_state()
    st.stop()

cols_to_show = [c for c in selected_cols if c in filtered.columns]
display_df = filtered[cols_to_show].head(EXPLORER_ROW_LIMIT).copy()

if "event_date" in display_df.columns:
    display_df["event_date"] = display_df["event_date"].dt.strftime("%Y-%m-%d")

# Rename to friendly names for display
display_df = friendly_columns(display_df)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Source": st.column_config.LinkColumn("Source"),
        "Tone": st.column_config.NumberColumn(format="%.1f"),
    },
)

if len(filtered) > EXPLORER_ROW_LIMIT:
    st.caption(
        f"Showing first {EXPLORER_ROW_LIMIT:,} of {len(filtered):,} rows. "
        "Narrow your filters to see more specific results."
    )

st.markdown("")
csv_data = friendly_columns(filtered[cols_to_show]).to_csv(index=False)
st.download_button(
    "Download filtered data",
    data=csv_data,
    file_name="gdelt_filtered.csv",
    mime="text/csv",
)
