"""
Page 3 — Event Types
One question: What kinds of events are happening, and where?
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
from src.config import COLOR_MAP_COUNTRY, COLOR_MAP_EVENT, COLOR_MAP_QUAD

try:
    df = load_events()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(df, key="et_countries")
df = apply_filters(df, countries)

st.header("Event Types")
st.caption("How events break down across the conflict–cooperation spectrum.")

if df.empty:
    empty_state()
    st.stop()

# ── Two main charts ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    quad_counts = df["QuadLabel"].value_counts().reset_index()
    quad_counts.columns = ["Event Class", "Events"]
    fig1 = px.pie(
        quad_counts, names="Event Class", values="Events",
        title="Event Classification",
        color="Event Class", color_discrete_map=COLOR_MAP_QUAD,
    )
    fig1.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    simple = df.groupby(["country", "EventType"]).size().reset_index(name="Events")
    simple = simple.rename(columns={"country": "Country", "EventType": "Type"})
    fig2 = px.bar(
        simple, x="Country", y="Events", color="Type",
        barmode="group", title="By Country",
        color_discrete_map=COLOR_MAP_EVENT,
    )
    fig2.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Top event families ────────────────────────────────────────────────────────
if "EventRootLabel" in df.columns:
    st.subheader("Most Common Event Families")
    root_counts = (
        df["EventRootLabel"]
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
    )
    root_counts.columns = ["Event Family", "Events"]
    fig3 = px.bar(
        root_counts, x="Events", y="Event Family", orientation="h",
        color="Events", color_continuous_scale="Blues",
    )
    fig3.update_layout(
        yaxis=dict(autorange="reversed"), height=380,
        margin=dict(t=20), coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Country summary inside expander ───────────────────────────────────────────
with st.expander("Conflict and cooperation ratios by country"):
    for country in countries:
        cdf = df[df["country"] == country]
        if cdf.empty:
            continue
        c_ratio = (cdf["EventType"] == "Conflict").mean()
        tone = cdf["AvgTone"].mean()
        st.markdown(
            f"**{country}** — {c_ratio:.0%} conflict, average tone {tone:.1f}"
        )
