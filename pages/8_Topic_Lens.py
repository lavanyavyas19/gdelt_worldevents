"""
Page 8 — Topic Lens
One question: What topics dominated the coverage?
"""

import streamlit as st
import pandas as pd
import plotly.express as px
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

top_n = st.sidebar.slider("Keywords to show", 5, 30, 15, key="tfidf_top_n")

st.header("Topic Lens")
st.caption("The most distinctive terms in event coverage, by country and during spikes.")

if df_filtered.empty:
    empty_state()
    st.stop()

# ── Top keywords by country ──────────────────────────────────────────────────
st.subheader("What topics appeared most")

country_kw = keywords_by_country(df_filtered, vectorizer, top_n=top_n)

if country_kw:
    tabs = st.tabs(list(country_kw.keys()))
    for tab, (country, kws) in zip(tabs, country_kw.items()):
        with tab:
            if kws:
                kw_df = pd.DataFrame(kws)
                kw_df = kw_df.rename(columns={"keyword": "Keyword", "score": "Relevance"})
                fig = px.bar(
                    kw_df, x="Relevance", y="Keyword", orientation="h",
                    color="Relevance", color_continuous_scale="Viridis",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=max(340, top_n * 24),
                    coloraxis_showscale=False,
                    margin=dict(t=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"No keywords found for {country}.")
else:
    st.caption("No keywords found.")

st.divider()

# ── Keywords during spikes ────────────────────────────────────────────────────
st.subheader("What stood out during spikes")
st.caption("Keywords extracted only from events on detected spike days.")

burst_kw = keywords_for_bursts(df_filtered, burst_df, vectorizer, top_n=top_n)

if burst_kw:
    tabs2 = st.tabs(list(burst_kw.keys()))
    for tab, (country, kws) in zip(tabs2, burst_kw.items()):
        with tab:
            if kws:
                kw_df = pd.DataFrame(kws)
                kw_df = kw_df.rename(columns={"keyword": "Keyword", "score": "Relevance"})
                fig = px.bar(
                    kw_df, x="Relevance", y="Keyword", orientation="h",
                    color="Relevance", color_continuous_scale="Magma",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=max(340, top_n * 24),
                    coloraxis_showscale=False,
                    margin=dict(t=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"No spike keywords for {country}.")
else:
    st.caption("No spikes detected — adjust the threshold on the Activity Spikes page.")

st.divider()

# ── How the story changed ────────────────────────────────────────────────────
st.subheader("How the story changed during spikes")
st.caption("Compare which terms dominate during normal days vs spike days.")

comparison = keywords_normal_vs_burst(df_filtered, burst_df, vectorizer, top_n=min(top_n, 10))

if comparison:
    for country, data in comparison.items():
        st.markdown(f"**{country}**")
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Normal days")
            if data["normal"]:
                ndf = pd.DataFrame(data["normal"]).rename(
                    columns={"keyword": "Keyword", "score": "Relevance"}
                )
                fig_n = px.bar(
                    ndf, x="Relevance", y="Keyword", orientation="h",
                    color_continuous_scale="Blues", color="Relevance",
                )
                fig_n.update_layout(
                    yaxis=dict(autorange="reversed"), height=280,
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(t=10),
                )
                st.plotly_chart(fig_n, use_container_width=True)
            else:
                st.caption("Not enough data.")

        with col2:
            st.caption("Spike days")
            if data["burst"]:
                bdf = pd.DataFrame(data["burst"]).rename(
                    columns={"keyword": "Keyword", "score": "Relevance"}
                )
                fig_b = px.bar(
                    bdf, x="Relevance", y="Keyword", orientation="h",
                    color_continuous_scale="Reds", color="Relevance",
                )
                fig_b.update_layout(
                    yaxis=dict(autorange="reversed"), height=280,
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(t=10),
                )
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.caption("Not enough data.")

        st.divider()
else:
    st.caption("This comparison requires detected spike days.")

# ── Download ──────────────────────────────────────────────────────────────────
with st.expander("Export keywords"):
    if country_kw:
        all_kws = []
        for country, kws in country_kw.items():
            for kw in kws:
                all_kws.append({"Country": country, "Keyword": kw["keyword"],
                                "Relevance": round(kw["score"], 4)})
        export_df = pd.DataFrame(all_kws)
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "Download keywords as CSV",
            data=csv_data,
            file_name="gdelt_keywords.csv",
            mime="text/csv",
        )
