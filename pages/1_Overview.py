"""
Page 1 — Overview
KPI summary cards: total events, events per country, average tone, burst count.
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
def load_events():
    return load_df(os.path.join(PROCESSED, "events"))


@st.cache_data
def load_bursts():
    return load_df(os.path.join(PROCESSED, "bursts"))


st.header("Overview")

try:
    df = load_events()
    burst_df = load_bursts()
except FileNotFoundError:
    st.error("Processed data not found. Run `python -m src.prepare_data` first.")
    st.stop()

# ── Sidebar filters ─────────────────────────────────────────────────────────
countries = st.sidebar.multiselect(
    "Countries", df["country"].unique().tolist(),
    default=df["country"].unique().tolist(),
)
df = df[df["country"].isin(countries)]
burst_df = burst_df[burst_df["country"].isin(countries)]

# ── KPI Row ─────────────────────────────────────────────────────────────────
total = len(df)
avg_tone = df["AvgTone"].mean()
n_bursts = int(burst_df["is_burst"].sum())
conflict_pct = (df["EventType"] == "Conflict").mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events", f"{total:,}")
c2.metric("Avg Tone", f"{avg_tone:.2f}")
c3.metric("Burst Days", n_bursts)
c4.metric("Conflict %", f"{conflict_pct:.1f}%")

st.divider()

# ── Events per country ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    country_counts = df["country"].value_counts().reset_index()
    country_counts.columns = ["country", "events"]
    fig1 = px.bar(
        country_counts, x="country", y="events", color="country",
        title="Events per Country",
        color_discrete_map={"USA": "#636EFA", "India": "#EF553B", "Iran": "#00CC96"},
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    type_counts = df.groupby(["country", "EventType"]).size().reset_index(name="count")
    fig2 = px.bar(
        type_counts, x="country", y="count", color="EventType",
        barmode="group", title="Conflict vs Cooperation per Country",
        color_discrete_map={"Conflict": "#EF553B", "Cooperation": "#00CC96"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tone distribution ───────────────────────────────────────────────────────
fig3 = px.histogram(
    df, x="AvgTone", color="country", nbins=50, marginal="box",
    title="Tone Distribution by Country", opacity=0.7,
)
st.plotly_chart(fig3, use_container_width=True)
