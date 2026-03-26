"""
Page 3 — Event Types
One question: What kinds of events are happening, and where?
Features: diverging bar chart, per-country event families, anomaly callout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# ── Diverging bar: cooperation ← | → conflict per country ────────────────────
st.subheader("Conflict vs Cooperation Balance")

ratios = []
for country in sorted(countries):
    cdf = df[df["country"] == country]
    if cdf.empty:
        continue
    total = len(cdf)
    coop_pct = (cdf["EventType"] == "Cooperation").mean() * 100
    conf_pct = (cdf["EventType"] == "Conflict").mean() * 100
    ratios.append({"Country": country, "Cooperation": coop_pct, "Conflict": -conf_pct})

if ratios:
    ratio_df = pd.DataFrame(ratios)

    fig_div = go.Figure()
    fig_div.add_trace(go.Bar(
        y=ratio_df["Country"], x=ratio_df["Cooperation"],
        name="Cooperation", orientation="h",
        marker_color="#00CC96",
        text=[f"{v:.0f}%" for v in ratio_df["Cooperation"]],
        textposition="inside",
    ))
    fig_div.add_trace(go.Bar(
        y=ratio_df["Country"], x=ratio_df["Conflict"],
        name="Conflict", orientation="h",
        marker_color="#EF553B",
        text=[f"{abs(v):.0f}%" for v in ratio_df["Conflict"]],
        textposition="inside",
    ))
    fig_div.update_layout(
        barmode="relative", height=200 + 50 * len(countries),
        margin=dict(t=20, b=20),
        xaxis_title="% of events",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_div.add_vline(x=0, line_color="black", line_width=1)
    st.plotly_chart(fig_div, use_container_width=True)

# ── Anomaly callout ──────────────────────────────────────────────────────────
if len(countries) > 1:
    overall_conflict = (df["EventType"] == "Conflict").mean()
    anomalies = []
    for country in countries:
        cdf = df[df["country"] == country]
        c_conflict = (cdf["EventType"] == "Conflict").mean()
        if c_conflict > overall_conflict + 0.1:
            anomalies.append(
                f"**{country}** has {c_conflict:.0%} conflict "
                f"(overall average: {overall_conflict:.0%})"
            )
    if anomalies:
        st.warning("Notable deviation: " + "; ".join(anomalies))

st.divider()

# ── Quad class breakdown ──────────────────────────────────────────────────────
st.subheader("Detailed Classification")

quad_by_country = (
    df.groupby(["country", "QuadLabel"]).size()
    .reset_index(name="Events")
    .rename(columns={"country": "Country", "QuadLabel": "Event Class"})
)
fig_quad = px.bar(
    quad_by_country, x="Country", y="Events", color="Event Class",
    barmode="group", color_discrete_map=COLOR_MAP_QUAD,
)
fig_quad.update_layout(height=380, margin=dict(t=20, b=20))
st.plotly_chart(fig_quad, use_container_width=True)

st.divider()

# ── Event families by country ────────────────────────────────────────────────
if "EventRootLabel" in df.columns:
    st.subheader("Event Families by Country")

    tabs = st.tabs(sorted(countries))
    for tab, country in zip(tabs, sorted(countries)):
        with tab:
            cdf = df[df["country"] == country]
            root_counts = (
                cdf["EventRootLabel"]
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
                margin=dict(t=10), coloraxis_showscale=False,
            )
            st.plotly_chart(fig3, use_container_width=True)
