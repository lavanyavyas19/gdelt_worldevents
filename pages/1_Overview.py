"""
Page 1 — Overview
One question: What's the big picture across these countries right now?
Features: KPI deltas, Key Insights section, event composition chart.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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

st.header("Overview")

if df.empty:
    empty_state()
    st.stop()

# ── Split data into halves for delta comparison ───────────────────────────────
date_min = df["event_date"].min()
date_max = df["event_date"].max()
date_mid = date_min + (date_max - date_min) / 2

first_half = df[df["event_date"] <= date_mid]
second_half = df[df["event_date"] > date_mid]

# ── KPI cards ─────────────────────────────────────────────────────────────────
total = len(df)
avg_tone = df["AvgTone"].mean()
n_bursts = int(burst_df["is_burst"].sum())
conflict_pct = (df["EventType"] == "Conflict").mean() * 100

first_total = len(first_half) if not first_half.empty else 0
second_total = len(second_half) if not second_half.empty else 0
events_delta = (second_total - first_total) / max(first_total, 1) * 100

first_tone = first_half["AvgTone"].mean() if not first_half.empty else 0
second_tone = second_half["AvgTone"].mean() if not second_half.empty else 0
tone_delta = second_tone - first_tone

first_conflict = (first_half["EventType"] == "Conflict").mean() * 100 if not first_half.empty else 0
second_conflict = (second_half["EventType"] == "Conflict").mean() * 100 if not second_half.empty else 0
conflict_delta = second_conflict - first_conflict

cols = st.columns(4)
with cols[0]:
    st.metric("Total Events", f"{total:,}", delta=f"{events_delta:+.0f}% vs prior half")
with cols[1]:
    st.metric("Avg Tone", tone_label(avg_tone), delta=f"{tone_delta:+.1f}")
with cols[2]:
    st.metric("Conflict Share", f"{conflict_pct:.0f}%", delta=f"{conflict_delta:+.1f}%")
with cols[3]:
    st.metric("Activity Spikes", str(n_bursts))

st.caption(f"{date_min.strftime('%b %d, %Y')} – {date_max.strftime('%b %d, %Y')}")

st.markdown("")

# ── Key Insights ──────────────────────────────────────────────────────────────
st.subheader("Key Insights")

# Per-country volume change
country_stats = {}
for c in countries:
    c_first = first_half[first_half["country"] == c]
    c_second = second_half[second_half["country"] == c]
    n_first = len(c_first)
    n_second = len(c_second)
    pct_change = (n_second - n_first) / max(n_first, 1) * 100
    n_spikes = int(burst_df[(burst_df["country"] == c) & burst_df["is_burst"]]["is_burst"].sum())
    c_tone = df[df["country"] == c]["AvgTone"].mean()
    country_stats[c] = {
        "total": len(df[df["country"] == c]),
        "pct_change": pct_change,
        "n_spikes": n_spikes,
        "tone": c_tone,
    }

# Find notable countries
highest_activity = max(country_stats, key=lambda c: country_stats[c]["total"]) if country_stats else None
most_spikes = max(country_stats, key=lambda c: country_stats[c]["n_spikes"]) if country_stats else None
biggest_shift = max(country_stats, key=lambda c: abs(country_stats[c]["pct_change"])) if country_stats else None

# Render insight cards in columns
insight_cols = st.columns(len(countries))
for col, country in zip(insight_cols, sorted(countries)):
    stats = country_stats.get(country, {})
    pct = stats.get("pct_change", 0)
    arrow = "▲" if pct > 0 else "▼"
    pct_color = "#EF553B" if pct > 10 else ("#00CC96" if pct < -10 else "#888888")
    spikes = stats.get("n_spikes", 0)
    tone_val = stats.get("tone", 0)
    total_c = stats.get("total", 0)

    with col:
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0; border-radius:8px;
                        padding:14px 16px; background:#fafafa;">
                <div style="font-size:1.0em; font-weight:600; margin-bottom:6px;">
                    {country}
                </div>
                <div style="font-size:0.85em; color:#444; line-height:1.7;">
                    <span style="color:{pct_color}; font-weight:bold;">
                        {arrow} {abs(pct):.0f}%
                    </span> vs first half<br>
                    <b>{total_c:,}</b> total events<br>
                    <b>{spikes}</b> spike {"day" if spikes == 1 else "days"}<br>
                    Tone: {tone_val:.2f} ({tone_label(tone_val)})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# ── Narrative insight box ─────────────────────────────────────────────────────
insights = []

if biggest_shift and abs(country_stats[biggest_shift]["pct_change"]) > 10:
    pct = country_stats[biggest_shift]["pct_change"]
    direction = "increased" if pct > 0 else "decreased"
    insights.append(
        f"**{biggest_shift}** saw the largest shift — events {direction} "
        f"by {abs(pct):.0f}% in the second half of the window."
    )

if highest_activity:
    insights.append(
        f"**{highest_activity}** accounts for the most total events "
        f"({country_stats[highest_activity]['total']:,})."
    )

if most_spikes and country_stats[most_spikes]["n_spikes"] > 0:
    insights.append(
        f"**{most_spikes}** recorded the most spike days "
        f"({country_stats[most_spikes]['n_spikes']})."
    )

if abs(conflict_delta) > 5:
    direction = "rising" if conflict_delta > 0 else "falling"
    insights.append(
        f"Conflict share is {direction} overall "
        f"({first_conflict:.0f}% → {second_conflict:.0f}%)."
    )

if abs(tone_delta) > 0.5:
    shift = "more negative" if tone_delta < 0 else "more positive"
    insights.append(f"Overall tone has shifted {shift} ({first_tone:.1f} → {second_tone:.1f}).")

if insights:
    st.info("  \n".join(f"• {i}" for i in insights))
else:
    st.info(
        f"Activity is stable across {len(countries)} countries "
        f"({total:,} events). Tone is {tone_label(avg_tone).lower()}."
    )

st.markdown("")

# ── Events by country (composition bar) ──────────────────────────────────────
type_counts = (
    df.groupby(["country", "EventType"])
    .size()
    .reset_index(name="Events")
    .rename(columns={"country": "Country", "EventType": "Type"})
)

fig = px.bar(
    type_counts, x="Country", y="Events", color="Type",
    barmode="stack", color_discrete_map=COLOR_MAP_EVENT,
    title="Events by Country",
)
fig.update_layout(
    margin=dict(t=40, b=20), height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ── Tone distribution (collapsible) ──────────────────────────────────────────
with st.expander("Tone distribution"):
    fig3 = px.histogram(
        df, x="AvgTone", color="country", nbins=50,
        opacity=0.7, color_discrete_map=COLOR_MAP_COUNTRY,
        labels={"AvgTone": "Tone", "country": "Country"},
    )
    fig3.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)
