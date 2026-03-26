"""
Page 8 — Characteristic Terms
One question: What metadata patterns dominate coverage, and how do they shift during spikes?
Features: per-country tabs, spike vs overall comparison, dynamic insight text.
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
    load_events, load_bursts, load_vectorizer,
    sidebar_country_filter, show_data_window,
    data_not_found, empty_state,
)
from src.keywords import (
    keywords_by_country, keywords_for_bursts, keywords_normal_vs_burst,
)

try:
    df = load_events()
    burst_df = load_bursts()
    vectorizer = load_vectorizer()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(df, key="tl_countries")
df_filtered = df[df["country"].isin(countries)]

top_n = st.sidebar.slider("Terms to show", 5, 25, 12, key="tfidf_top_n")

st.header("Characteristic Terms")
st.caption(
    "The most distinctive actor, location, and event-type patterns by country — "
    "and how they change during activity spikes. "
    "Derived from structured GDELT metadata (actors, locations, event codes)."
)

if df_filtered.empty:
    empty_state()
    st.stop()

# ── Pre-compute all term data (outside tabs for efficiency) ──────────────────
country_kw = keywords_by_country(df_filtered, vectorizer, top_n=top_n)
comparison = keywords_normal_vs_burst(df_filtered, burst_df, vectorizer, top_n=min(top_n, 12))
burst_kw = keywords_for_bursts(df_filtered, burst_df, vectorizer, top_n=top_n)

tab_countries = sorted(c for c in countries if c in country_kw or c in comparison)

if not tab_countries:
    empty_state("No term data available for selected countries.")
    st.stop()

# ── One tab per country ───────────────────────────────────────────────────────
tabs = st.tabs(tab_countries)

for tab, country in zip(tabs, tab_countries):
    with tab:

        # ── Section 1: Top Keywords Overall ──────────────────────────────────
        st.subheader("Top Keywords Overall")
        kws = country_kw.get(country, [])

        if kws:
            kw_df = pd.DataFrame(kws).rename(
                columns={"keyword": "Term", "score": "Distinctiveness"}
            )
            fig_overall = px.bar(
                kw_df, x="Distinctiveness", y="Term", orientation="h",
                color="Distinctiveness", color_continuous_scale="Blues",
            )
            fig_overall.update_layout(
                yaxis=dict(autorange="reversed"),
                height=max(300, len(kws) * 26),
                coloraxis_showscale=False,
                margin=dict(t=10, l=10),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_overall.update_xaxes(showgrid=False)
            fig_overall.update_yaxes(gridcolor="rgba(0,0,0,0.04)")
            st.plotly_chart(fig_overall, use_container_width=True)
        else:
            st.caption(f"No terms found for {country}.")

        st.divider()

        # ── Section 2: Top Keywords During Spikes ────────────────────────────
        st.subheader("Top Keywords During Spikes")
        st.caption(
            "Terms that are more or less distinctive during spike days "
            "compared to normal activity periods."
        )

        data = comparison.get(country, {})
        burst_kws_dict = {kw["keyword"]: kw["score"] for kw in data.get("burst", [])}
        normal_kws_dict = {kw["keyword"]: kw["score"] for kw in data.get("normal", [])}
        all_terms = set(burst_kws_dict.keys()) | set(normal_kws_dict.keys())

        if all_terms:
            delta_data = []
            for term in all_terms:
                b_score = burst_kws_dict.get(term, 0)
                n_score = normal_kws_dict.get(term, 0)
                delta = b_score - n_score
                delta_data.append({
                    "Term": term,
                    "Delta": delta,
                    "Direction": "Higher during spikes" if delta > 0 else "Lower during spikes",
                })

            delta_df = pd.DataFrame(delta_data).sort_values("Delta")
            # Keep top and bottom 8 for a balanced diverging view
            if len(delta_df) > 16:
                delta_df = pd.concat([delta_df.head(8), delta_df.tail(8)])

            fig_delta = px.bar(
                delta_df, x="Delta", y="Term", orientation="h",
                color="Direction",
                color_discrete_map={
                    "Higher during spikes": "#EF553B",
                    "Lower during spikes": "#636EFA",
                },
            )
            fig_delta.add_vline(x=0, line_color="#CCCCCC", line_width=1)
            fig_delta.update_layout(
                height=max(300, len(delta_df) * 28),
                margin=dict(t=10, l=10),
                xaxis_title="Change in distinctiveness (spike vs normal)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_delta.update_xaxes(showgrid=False)
            fig_delta.update_yaxes(gridcolor="rgba(0,0,0,0.04)")
            st.plotly_chart(fig_delta, use_container_width=True)

            # ── Dynamic insight ───────────────────────────────────────────────
            spike_terms = [
                row["Term"]
                for _, row in delta_df.sort_values("Delta", ascending=False).iterrows()
                if row["Direction"] == "Higher during spikes"
            ]
            overall_top = [kw["keyword"] for kw in kws[:3]] if kws else []
            spike_unique = [t for t in spike_terms[:3] if t not in overall_top]

            if spike_unique:
                st.info(
                    f"During spike periods, **{country}** coverage shifts toward "
                    f"**{spike_unique[0]}**"
                    + (f" and **{spike_unique[1]}**" if len(spike_unique) > 1 else "")
                    + " — terms not prominent in day-to-day activity, "
                    "suggesting heightened geopolitical or crisis-related coverage."
                )
            elif spike_terms:
                st.info(
                    f"Spike periods in **{country}** reinforce existing patterns — "
                    f"**{spike_terms[0]}** remains the dominant theme across both normal "
                    "and elevated activity days."
                )
            else:
                st.info(
                    f"Insufficient spike data for **{country}** to identify "
                    "meaningful pattern shifts."
                )
        else:
            st.caption("Not enough spike data for comparison. Try lowering the detection threshold.")

# ── Methodology note ─────────────────────────────────────────────────────────
with st.expander("About this analysis"):
    st.markdown("""
**What this measures:** TF-IDF distinctiveness scores across structured GDELT metadata
fields — actor names, locations, event family labels, and CAMEO classification codes.

**What this does NOT measure:** Actual article text or topics. GDELT v1 does not include
article content. Results reflect which *metadata patterns* are most distinctive per
country, not which *topics* dominate coverage.

**Methodology:** Unigram and bigram TF-IDF with sublinear term frequency (log-scaled),
custom geographic stopwords, and country-specific name removal. 800 features extracted.

**For true topic analysis:** Consider integrating GDELT v2's Global Knowledge Graph (GKG),
which includes themes, organisations, and topics extracted directly from news articles.
    """)
