"""
Page 6 — Burst Dashboard
One question: When did something unusual happen?
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, load_bursts, sidebar_country_filter,
    show_data_window, data_not_found, empty_state, metric_row,
    friendly_columns,
)
from src.config import COLOR_MAP_COUNTRY, BURST_ROLLING_WINDOW, BURST_Z_THRESHOLD
from src.burst import detect_bursts, get_burst_summary

try:
    df = load_events()
    default_burst_df = load_bursts()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(default_burst_df, key="burst_countries")

st.sidebar.divider()
st.sidebar.subheader("Sensitivity")
z_threshold = st.sidebar.slider(
    "Detection threshold", 1.0, 5.0, BURST_Z_THRESHOLD, 0.25, key="burst_z",
    help="Lower = more sensitive, higher = only the strongest spikes",
)
rolling_window = st.sidebar.slider(
    "Baseline window (days)", 3, 14, BURST_ROLLING_WINDOW, key="burst_win",
    help="How many days to average when calculating the 'typical' level",
)
recompute = st.sidebar.button("Recalculate", key="burst_recompute")

st.header("Activity Spikes")
st.caption("Days when event counts jumped well above the typical level.")

if recompute:
    with st.spinner("Recalculating…"):
        burst_df = detect_bursts(df, rolling_window=rolling_window, z_threshold=z_threshold)
else:
    burst_df = default_burst_df.copy()

burst_df = burst_df[burst_df["country"].isin(countries)]

if burst_df.empty:
    empty_state("No data for selected countries.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_burst = int(burst_df["is_burst"].sum())
max_z = burst_df["z_score"].max()
avg_burst_count = (
    burst_df[burst_df["is_burst"]]["event_count"].mean()
    if total_burst > 0 else 0
)

metric_row([
    ("Spike Days", str(total_burst)),
    ("Days Analysed", str(burst_df["day"].nunique())),
    ("Strongest Spike", f"{max_z:.1f}x"),
    ("Avg Events on Spike Day", f"{avg_burst_count:,.0f}" if total_burst > 0 else "—"),
])

st.markdown("")

# ── Timeline per country ─────────────────────────────────────────────────────
for country in sorted(countries):
    cdf = burst_df[burst_df["country"] == country].copy().sort_values("day")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cdf["day"], y=cdf["event_count"],
        mode="lines+markers", name="Events",
        line=dict(color=COLOR_MAP_COUNTRY.get(country, "#636EFA"), width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=cdf["day"], y=cdf["rolling_mean"],
        mode="lines", name="Typical Level",
        line=dict(color="#AB63FA", dash="dash", width=1.5),
    ))
    bursts = cdf[cdf["is_burst"]]
    if not bursts.empty:
        fig.add_trace(go.Scatter(
            x=bursts["day"], y=bursts["event_count"],
            mode="markers", name="Spike",
            marker=dict(color="red", size=14, symbol="star"),
        ))
    fig.update_layout(
        title=f"{country}", height=370,
        xaxis_title="", yaxis_title="Events",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Event types during spikes ─────────────────────────────────────────────────
st.subheader("What Happened During Spikes")
burst_days_set = burst_df[burst_df["is_burst"]][["day", "country"]]
if not burst_days_set.empty:
    burst_events = df.merge(burst_days_set, on=["day", "country"], how="inner")
    if not burst_events.empty and "QuadLabel" in burst_events.columns:
        quad_burst = burst_events.groupby(["country", "QuadLabel"]).size().reset_index(name="Events")
        quad_burst = quad_burst.rename(columns={"country": "Country", "QuadLabel": "Event Class"})
        fig_q = px.bar(
            quad_burst, x="Country", y="Events", color="Event Class",
            barmode="stack",
        )
        fig_q.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig_q, use_container_width=True)
else:
    st.caption("No spikes detected with current settings.")

# ── Detailed tables inside expander ───────────────────────────────────────────
with st.expander("Spike day details"):
    burst_summary = get_burst_summary(burst_df)
    if not burst_summary.empty:
        display = burst_summary.copy()
        display["day"] = display["day"].dt.strftime("%Y-%m-%d")
        display = display.rename(columns={
            "day": "Date", "country": "Country", "event_count": "Events",
            "rolling_mean": "Typical Level", "z_score": "Burst Strength",
        })
        display["Typical Level"] = display["Typical Level"].round(0).astype(int)
        display["Burst Strength"] = display["Burst Strength"].round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("No spike days to show. Try lowering the detection threshold.")

with st.expander("Burst strength comparison"):
    fig_z = px.line(
        burst_df, x="day", y="z_score", color="country",
        color_discrete_map=COLOR_MAP_COUNTRY,
        labels={"day": "Date", "z_score": "Burst Strength", "country": "Country"},
    )
    fig_z.add_hline(
        y=z_threshold, line_dash="dash", line_color="red",
        annotation_text=f"Threshold ({z_threshold})",
    )
    fig_z.update_layout(height=320, margin=dict(t=20))
    st.plotly_chart(fig_z, use_container_width=True)
