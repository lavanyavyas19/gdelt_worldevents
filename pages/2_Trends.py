"""
Page 2 — Trends
One question: How is activity changing over time?
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, sidebar_country_filter, show_data_window,
    apply_filters, data_not_found, empty_state,
)
from src.config import COLOR_MAP_COUNTRY
from src.aggregation import aggregate_by

try:
    df = load_events()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(df, key="tr_countries")
granularity = st.sidebar.radio(
    "Time scale", ["Daily", "Weekly", "Monthly"], index=2, key="tr_gran"
)
normalise = st.sidebar.toggle("Show as ratios", value=False, key="tr_norm")

df = apply_filters(df, countries)

st.header("Trends")
st.caption("How event volume and tone shift over the analysis window.")

if df.empty:
    empty_state()
    st.stop()

# ── Aggregate ─────────────────────────────────────────────────────────────────
gran_map = {"Daily": "day", "Weekly": "week_label", "Monthly": "month_label"}
gran_col = gran_map[granularity]

if gran_col == "day":
    df = df.copy()
    df["day_str"] = df["day"].dt.strftime("%Y-%m-%d")
    agg = aggregate_by(df, "day_str")
else:
    agg = aggregate_by(df, gran_col)

if agg.empty:
    empty_state("Not enough data to show trends.")
    st.stop()

# ── Event volume ──────────────────────────────────────────────────────────────
y_col = "cooperation_ratio" if normalise else "total_events"
y_label = "Cooperation Share" if normalise else "Events"

fig1 = px.line(
    agg, x="period", y=y_col, color="country", markers=True,
    color_discrete_map=COLOR_MAP_COUNTRY,
    labels={"period": "Period", y_col: y_label, "country": "Country"},
)
fig1.update_layout(title=f"Event Volume ({granularity})", height=400, margin=dict(t=40))
st.plotly_chart(fig1, use_container_width=True)

# ── Tone over time ────────────────────────────────────────────────────────────
fig2 = px.line(
    agg, x="period", y="avg_tone", color="country", markers=True,
    color_discrete_map=COLOR_MAP_COUNTRY,
    labels={"period": "Period", "avg_tone": "Tone", "country": "Country"},
)
fig2.update_layout(title=f"Average Tone ({granularity})", height=380, margin=dict(t=40))
st.plotly_chart(fig2, use_container_width=True)

# ── Composition (collapsible) ─────────────────────────────────────────────────
with st.expander("Event type composition over time"):
    melted = agg.melt(
        id_vars=["period", "country"],
        value_vars=["conflict_events", "cooperation_events"],
        var_name="type", value_name="count",
    )
    melted["type"] = melted["type"].map({
        "conflict_events": "Conflict",
        "cooperation_events": "Cooperation",
    })
    fig3 = px.area(
        melted, x="period", y="count", color="type",
        facet_col="country",
        color_discrete_map={"Conflict": "#EF553B", "Cooperation": "#00CC96"},
        labels={"period": "Period", "count": "Events", "type": "Type"},
    )
    fig3.update_layout(height=320, margin=dict(t=30))
    st.plotly_chart(fig3, use_container_width=True)
