"""
Page 6 — Activity Spikes
One question: When did something unusual happen?
Features: tabbed layout, confidence bands, spike drill-down, cross-country correlation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, load_bursts, sidebar_country_filter,
    show_data_window, data_not_found, empty_state, metric_row,
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
    help="Lower = more sensitive. Higher = only the strongest spikes.",
)
rolling_window = st.sidebar.slider(
    "Baseline window (days)", 3, 14, BURST_ROLLING_WINDOW, key="burst_win",
    help="Days averaged to establish the 'typical' activity level.",
)
recompute = st.sidebar.button("Recalculate", key="burst_recompute")

st.header("Activity Spikes")
st.caption("Days when event counts rose significantly above the established baseline.")

if recompute:
    with st.spinner("Recalculating…"):
        burst_df = detect_bursts(df, rolling_window=rolling_window, z_threshold=z_threshold)
else:
    burst_df = default_burst_df.copy()

burst_df = burst_df[burst_df["country"].isin(countries)]

if burst_df.empty:
    empty_state("No data for selected countries.")
    st.stop()

# ── Derived values used across tabs ──────────────────────────────────────────
total_burst = int(burst_df["is_burst"].sum())

# Guard against NaN / Inf in z_score before computing max
_z_clean = burst_df["z_score"].replace([np.inf, -np.inf], np.nan)
max_z = _z_clean.max()  # NaN when no valid scores exist
max_z_display = f"{max_z:.1f}σ" if pd.notna(max_z) else "—"

avg_burst_count = (
    burst_df[burst_df["is_burst"]]["event_count"].mean()
    if total_burst > 0 else 0
)
burst_summary = get_burst_summary(burst_df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_summary, tab_countries, tab_details = st.tabs([
    "Summary", "Country Analysis", "Spike Details"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Summary
# ════════════════════════════════════════════════════════════════════════════
with tab_summary:
    # KPI row
    metric_row([
        ("Spike Days", str(total_burst)),
        ("Days Analysed", str(burst_df["day"].nunique())),
        ("Strongest Spike", max_z_display),
        ("Avg Events on Spike Day", f"{avg_burst_count:,.0f}" if total_burst > 0 else "—"),
    ])
    st.markdown("")

    # Cross-country simultaneous spike analysis
    if len(countries) > 1 and total_burst > 0:
        st.subheader("Cross-Country Spike Patterns")

        burst_days_per_country = {
            c: set(
                burst_df[(burst_df["country"] == c) & burst_df["is_burst"]]["day"]
                .dt.strftime("%Y-%m-%d")
            )
            for c in countries
        }

        all_burst_days = set()
        for days in burst_days_per_country.values():
            all_burst_days |= days

        overlaps = [
            {"Date": day, "Countries": ", ".join(
                c for c, days in burst_days_per_country.items() if day in days
            )}
            for day in sorted(all_burst_days)
            if sum(1 for days in burst_days_per_country.values() if day in days) > 1
        ]

        if overlaps:
            st.warning(
                f"{len(overlaps)} day{'s' if len(overlaps) > 1 else ''} with simultaneous "
                f"spikes across countries — possible shared triggers."
            )
            st.dataframe(pd.DataFrame(overlaps), use_container_width=True, hide_index=True)
        else:
            st.caption("No simultaneous spikes detected across countries in this window.")

    # Event types during spikes
    st.markdown("")
    st.subheader("Event Composition During Spikes")
    burst_days_set = burst_df[burst_df["is_burst"]][["day", "country"]]
    if not burst_days_set.empty:
        burst_events = df.merge(burst_days_set, on=["day", "country"], how="inner")
        if not burst_events.empty and "QuadLabel" in burst_events.columns:
            quad_burst = (
                burst_events.groupby(["country", "QuadLabel"])
                .size().reset_index(name="Events")
                .rename(columns={"country": "Country", "QuadLabel": "Event Class"})
            )
            fig_q = px.bar(
                quad_burst, x="Country", y="Events", color="Event Class",
                barmode="stack",
                color_discrete_map={
                    "Verbal Cooperation": "#00CC96",
                    "Material Cooperation": "#19D3F3",
                    "Verbal Conflict": "#FFA15A",
                    "Material Conflict": "#EF553B",
                },
            )
            fig_q.update_layout(
                height=340, margin=dict(t=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.caption("No spike event data available.")
    else:
        st.caption("No spike days detected with current settings.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Country Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_countries:
    for country in sorted(countries):
        cdf = burst_df[burst_df["country"] == country].copy().sort_values("day")
        if cdf.empty:
            continue

        fig = go.Figure()

        # Confidence band (±2 std) — low opacity fill
        upper = cdf["rolling_mean"] + 2 * cdf["rolling_std"]
        lower = (cdf["rolling_mean"] - 2 * cdf["rolling_std"]).clip(lower=0)

        fig.add_trace(go.Scatter(
            x=cdf["day"], y=upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=cdf["day"], y=lower,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(171, 99, 250, 0.07)",
            showlegend=False, hoverinfo="skip",
            name="Expected range",
        ))

        # Daily event count line
        fig.add_trace(go.Scatter(
            x=cdf["day"], y=cdf["event_count"],
            mode="lines+markers", name="Events",
            line=dict(color=COLOR_MAP_COUNTRY.get(country, "#636EFA"), width=2),
            marker=dict(size=3),
        ))

        # Rolling mean (baseline)
        fig.add_trace(go.Scatter(
            x=cdf["day"], y=cdf["rolling_mean"],
            mode="lines", name="Baseline",
            line=dict(color="#AB63FA", dash="dash", width=1.5),
        ))

        # Spike markers — small red dots, no emoji
        bursts = cdf[cdf["is_burst"]]
        if not bursts.empty:
            fig.add_trace(go.Scatter(
                x=bursts["day"], y=bursts["event_count"],
                mode="markers", name="Spike",
                marker=dict(
                    color="red", size=8, symbol="circle",
                    line=dict(width=1.5, color="darkred"),
                ),
                customdata=np.column_stack([
                    bursts["day"].dt.strftime("%Y-%m-%d"),
                    bursts["z_score"].round(1),
                ]),
                hovertemplate="%{customdata[0]}  ·  %{y:,} events  ·  %{customdata[1]}σ<extra>Spike</extra>",
            ))

        fig.update_layout(
            title=country,
            height=340,
            xaxis_title="", yaxis_title="Events",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=45, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")

        st.plotly_chart(fig, use_container_width=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spike Details
# ════════════════════════════════════════════════════════════════════════════
with tab_details:
    if burst_summary.empty:
        st.caption("No spike days detected. Try lowering the detection threshold.")
    else:
        # Drill-down links
        st.subheader("Spike Days")
        st.caption("Select a spike day to investigate in the Event Chain Explorer.")

        for _, row in burst_summary.iterrows():
            day_str = row["day"].strftime("%Y-%m-%d")
            country = row["country"]
            z = row["z_score"]
            events = int(row["event_count"])
            severity = "High" if z >= 3 else "Moderate"
            severity_color = "#EF553B" if z >= 3 else "#FFA15A"

            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"**{day_str}** · {country} · {events:,} events · "
                    f"{z:.1f}σ · "
                    f"<span style='color:{severity_color};font-weight:600;'>"
                    f"{severity}</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button(
                    "Investigate →",
                    key=f"spike_btn_{day_str}_{country}",
                    help=f"Open Event Chain for {country} on {day_str}",
                ):
                    st.session_state["selected_date"]    = day_str
                    st.session_state["selected_country"] = country
                    st.switch_page("pages/7_Event_Chain.py")

        st.markdown("")
        st.divider()

        # Spike details table
        st.subheader("Spike Day Details")
        display = burst_summary.copy()
        display["day"] = display["day"].dt.strftime("%Y-%m-%d")
        display["Severity"] = display["z_score"].apply(
            lambda z: "High" if z >= 3 else "Moderate"
        )
        display = display.rename(columns={
            "day": "Date", "country": "Country",
            "event_count": "Events",
            "rolling_mean": "Baseline (avg)",
            "z_score": "Strength (σ)",
        })
        display["Baseline (avg)"] = (
            pd.to_numeric(display["Baseline (avg)"], errors="coerce")
            .fillna(0).round(0).astype(int)
        )
        display["Strength (σ)"] = display["Strength (σ)"].round(1)

        # Highlight high-severity rows
        st.dataframe(
            display[["Date", "Country", "Events", "Baseline (avg)", "Strength (σ)", "Severity"]],
            use_container_width=True,
            hide_index=True,
        )
