"""
Page 1 — Overview
One question: What's the big picture across these countries right now?
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, load_bursts, sidebar_country_filter,
    show_data_window, apply_filters, data_not_found,
    metric_row, empty_state, tone_label,
)
from src.config import COLOR_MAP_COUNTRY, COLOR_MAP_EVENT

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = load_events()
    burst_df = load_bursts()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(df, key="ov_countries")
df = apply_filters(df, countries)
burst_df = burst_df[burst_df["country"].isin(countries)]

# ── Header ────────────────────────────────────────────────────────────────────
st.header("Overview")

if df.empty:
    empty_state()
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
total = len(df)
avg_tone = df["AvgTone"].mean()
n_bursts = int(burst_df["is_burst"].sum())
conflict_pct = (df["EventType"] == "Conflict").mean() * 100

metric_row([
    ("Total Events", f"{total:,}"),
    ("Countries", str(len(countries))),
    ("Average Tone", tone_label(avg_tone)),
    ("Conflict Share", f"{conflict_pct:.0f}%"),
])

date_min = df["event_date"].min().strftime("%b %d, %Y")
date_max = df["event_date"].max().strftime("%b %d, %Y")
st.caption(f"{date_min} – {date_max}")

st.markdown("")

# ── Two main charts ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    country_counts = df["country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Events"]
    fig1 = px.bar(
        country_counts, x="Country", y="Events", color="Country",
        color_discrete_map=COLOR_MAP_COUNTRY,
    )
    fig1.update_layout(
        title="Events by Country", showlegend=False,
        margin=dict(t=40, b=20), height=360,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    type_counts = df.groupby(["country", "EventType"]).size().reset_index(name="Events")
    type_counts = type_counts.rename(columns={"country": "Country", "EventType": "Type"})
    fig2 = px.bar(
        type_counts, x="Country", y="Events", color="Type",
        barmode="stack", color_discrete_map=COLOR_MAP_EVENT,
    )
    fig2.update_layout(
        title="Conflict vs Cooperation", margin=dict(t=40, b=20), height=360,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Insight box ───────────────────────────────────────────────────────────────
most_events_country = country_counts.iloc[0]["Country"] if not country_counts.empty else "N/A"
tone_lbl = "negative" if avg_tone < -0.3 else ("neutral" if avg_tone <= 0.3 else "positive")

st.markdown(
    f"Most events are concentrated in **{most_events_country}**. "
    f"Overall tone is **{tone_lbl}** ({avg_tone:.1f}). "
    f"**{conflict_pct:.0f}%** of events involve conflict. "
    f"**{n_bursts}** days had unusual activity spikes."
)

# ── Detailed tables inside expander ───────────────────────────────────────────
with st.expander("Top event locations"):
    top_locs = (
        df.groupby(["ActionGeo_FullName", "country"])
        .agg(Events=("GLOBALEVENTID", "count"), Tone=("AvgTone", "mean"))
        .reset_index()
        .rename(columns={"ActionGeo_FullName": "Location", "country": "Country"})
        .nlargest(10, "Events")
    )
    top_locs["Tone"] = top_locs["Tone"].round(1)
    st.dataframe(top_locs, use_container_width=True, hide_index=True)

with st.expander("Tone distribution"):
    fig3 = px.histogram(
        df, x="AvgTone", color="country", nbins=50,
        opacity=0.7, color_discrete_map=COLOR_MAP_COUNTRY,
        labels={"AvgTone": "Tone", "country": "Country"},
    )
    fig3.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)
