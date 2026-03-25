"""
Page 2 — Trends
Event count and sentiment over time (monthly / weekly toggle).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.storage import load_df
PROCESSED = os.path.join(ROOT, "data", "processed")


@st.cache_data
def load_agg(granularity):
    fname = "agg_monthly.parquet" if granularity == "Monthly" else "agg_weekly.parquet"
    return load_df(os.path.join(PROCESSED, fname.replace(".parquet", "")))


st.header("Trends")

granularity = st.sidebar.radio("Granularity", ["Monthly", "Weekly"], index=0)
try:
    agg = load_agg(granularity)
except FileNotFoundError:
    st.error("Aggregated data not found. Run the pipeline first.")
    st.stop()

countries = st.sidebar.multiselect(
    "Countries", agg["country"].unique().tolist(),
    default=agg["country"].unique().tolist(), key="trend_countries",
)
agg = agg[agg["country"].isin(countries)]

# Toggle: absolute vs normalised
normalise = st.sidebar.toggle("Normalised metrics", value=False)
y_col = "cooperation_ratio" if normalise else "total_events"

# ── Event count over time ───────────────────────────────────────────────────
fig1 = px.line(
    agg, x="period", y="total_events" if not normalise else "cooperation_ratio",
    color="country", markers=True,
    title=f"{'Cooperation Ratio' if normalise else 'Event Count'} Over Time",
    color_discrete_map={"USA": "#636EFA", "India": "#EF553B", "Iran": "#00CC96"},
)
st.plotly_chart(fig1, use_container_width=True)

# ── Average tone over time ─────────────────────────────────────────────────
fig2 = px.line(
    agg, x="period", y="avg_tone", color="country", markers=True,
    title="Average Tone Over Time",
    color_discrete_map={"USA": "#636EFA", "India": "#EF553B", "Iran": "#00CC96"},
)
st.plotly_chart(fig2, use_container_width=True)

# ── Conflict vs Cooperation stacked area ────────────────────────────────────
melted = agg.melt(
    id_vars=["period", "country"],
    value_vars=["conflict_events", "cooperation_events"],
    var_name="type", value_name="count",
)
fig3 = px.area(
    melted, x="period", y="count", color="type",
    facet_col="country", title="Conflict & Cooperation Events",
    color_discrete_map={
        "conflict_events": "#EF553B",
        "cooperation_events": "#00CC96",
    },
)
st.plotly_chart(fig3, use_container_width=True)
