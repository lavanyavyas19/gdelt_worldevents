"""
Page 2 — Trends
One question: How is activity changing over time?
Features: weekly aggregation, burst-week markers, trend indicators.
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
normalise = st.sidebar.toggle(
    "Show cooperation share",
    value=False, key="tr_norm",
    help="Switch y-axis to cooperation share (0–1 ratio) instead of raw event count.",
)

df = apply_filters(df, countries)

st.header("Trends")
st.caption("Weekly patterns in event volume and tone across the analysis window.")

if df.empty:
    empty_state()
    st.stop()

# ── Half-period trend indicators ─────────────────────────────────────────────
date_mid = df["event_date"].min() + (df["event_date"].max() - df["event_date"].min()) / 2
trend_cols = st.columns(len(countries))
for tcol, country in zip(trend_cols, sorted(countries)):
    with tcol:
        cdf = df[df["country"] == country]
        first = cdf[cdf["event_date"] <= date_mid]
        second = cdf[cdf["event_date"] > date_mid]
        if len(first) > 0:
            pct = (len(second) - len(first)) / len(first) * 100
            arrow = "▲" if pct > 0 else "▼"
            color = "#EF553B" if pct > 10 else ("#00CC96" if pct < -10 else "#888")
            st.markdown(
                f"**{country}** "
                f"<span style='color:{color};font-weight:bold;'>{arrow} {abs(pct):.0f}%</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{country}**")

st.markdown("")

# ── Weekly aggregation ────────────────────────────────────────────────────────
agg = aggregate_by(df, "week_label")

if agg.empty:
    empty_state("Not enough data to show trends.")
    st.stop()

# ── Event volume chart ────────────────────────────────────────────────────────
y_col = "cooperation_ratio" if normalise else "total_events"
y_label = "Cooperation Share" if normalise else "Events per Week"

fig1 = px.line(
    agg, x="period", y=y_col, color="country", markers=True,
    color_discrete_map=COLOR_MAP_COUNTRY,
    labels={"period": "Week", y_col: y_label, "country": "Country"},
)

fig1.update_layout(
    title="Event Volume (Weekly)",
    height=420, margin=dict(t=45, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
    yaxis_title=y_label,
)
st.plotly_chart(fig1, use_container_width=True)

# ── Tone over time ────────────────────────────────────────────────────────────
fig2 = px.line(
    agg, x="period", y="avg_tone", color="country", markers=True,
    color_discrete_map=COLOR_MAP_COUNTRY,
    labels={"period": "Week", "avg_tone": "Avg Tone", "country": "Country"},
)
fig2.add_hline(y=0, line_dash="dot", line_color="#CCCCCC", opacity=0.6)

fig2.update_layout(
    title="Average Tone (Weekly)",
    height=380, margin=dict(t=45, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Summary caption ───────────────────────────────────────────────────────────
total_first = len(df[df["event_date"] <= date_mid])
total_second = len(df[df["event_date"] > date_mid])
overall_pct = (total_second - total_first) / max(total_first, 1) * 100
direction = "increasing" if overall_pct > 5 else ("decreasing" if overall_pct < -5 else "stable")

first_tone = df[df["event_date"] <= date_mid]["AvgTone"].mean()
second_tone = df[df["event_date"] > date_mid]["AvgTone"].mean()
tone_shift = (
    "more negative" if second_tone < first_tone - 0.3
    else ("more positive" if second_tone > first_tone + 0.3 else "stable")
)
st.caption(
    f"Overall activity is {direction} ({overall_pct:+.0f}%). "
    f"Tone has shifted {tone_shift} ({first_tone:.1f} → {second_tone:.1f})."
)

