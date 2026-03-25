"""
Page 6 — Burst Dashboard
Highlight days with unusual event spikes (z-score based detection).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.storage import load_df
PROCESSED = os.path.join(ROOT, "data", "processed")


@st.cache_data
def load_bursts():
    return load_df(os.path.join(PROCESSED, "bursts"))


@st.cache_data
def load_events():
    return load_df(os.path.join(PROCESSED, "events"))


st.header("Burst Detection Dashboard")

try:
    burst_df = load_bursts()
    df = load_events()
except FileNotFoundError:
    st.error("Processed data not found."); st.stop()

countries = st.sidebar.multiselect(
    "Countries", burst_df["country"].unique().tolist(),
    default=burst_df["country"].unique().tolist(), key="burst_countries",
)
burst_df = burst_df[burst_df["country"].isin(countries)]

# ── KPI ─────────────────────────────────────────────────────────────────────
total_burst = int(burst_df["is_burst"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Total Burst Days", total_burst)
c2.metric("Max Z-Score", f"{burst_df['z_score'].max():.2f}")
c3.metric("Avg Events on Burst Days",
          f"{burst_df[burst_df['is_burst']]['event_count'].mean():.0f}"
          if total_burst > 0 else "N/A")

st.divider()

# ── Timeline with burst highlights ──────────────────────────────────────────
for country in countries:
    cdf = burst_df[burst_df["country"] == country].copy()
    cdf["day"] = pd.to_datetime(cdf["day"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cdf["day"], y=cdf["event_count"],
        mode="lines+markers", name="Event Count",
        line=dict(color="#636EFA"),
    ))
    fig.add_trace(go.Scatter(
        x=cdf["day"], y=cdf["rolling_mean"],
        mode="lines", name="Rolling Mean",
        line=dict(color="#AB63FA", dash="dash"),
    ))

    # Highlight bursts
    bursts = cdf[cdf["is_burst"]]
    if not bursts.empty:
        fig.add_trace(go.Scatter(
            x=bursts["day"], y=bursts["event_count"],
            mode="markers", name="BURST",
            marker=dict(color="red", size=12, symbol="star"),
        ))

    fig.update_layout(
        title=f"{country} — Daily Events & Burst Detection",
        xaxis_title="Date", yaxis_title="Event Count",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Z-score heatmap ─────────────────────────────────────────────────────────
st.subheader("Z-Score Details")
pivot = burst_df.pivot_table(index="country", columns="day", values="z_score")
if not pivot.empty:
    fig_heat = px.imshow(
        pivot, aspect="auto", color_continuous_scale="RdYlGn_r",
        title="Z-Score Heatmap (Country x Day)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Burst event details ────────────────────────────────────────────────────
st.subheader("Burst Day Details")
burst_days = burst_df[burst_df["is_burst"]][["day", "country", "event_count", "z_score"]]
st.dataframe(burst_days.sort_values("z_score", ascending=False), use_container_width=True)
