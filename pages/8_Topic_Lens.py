"""
Page 8 — Topic Lens
TF-IDF keyword extraction per country and for burst periods.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.storage import load_df

from src.tfidf_module import keywords_by_country, keywords_for_bursts

PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")


@st.cache_data
def load_events():
    return load_df(os.path.join(PROCESSED, "events"))


@st.cache_data
def load_bursts():
    return load_df(os.path.join(PROCESSED, "bursts"))


@st.cache_resource
def load_vectorizer():
    with open(os.path.join(MODELS, "tfidf_vectorizer.pkl"), "rb") as f:
        return pickle.load(f)


st.header("Topic Lens — Keyword Intelligence")

try:
    df = load_events()
    burst_df = load_bursts()
    vectorizer = load_vectorizer()
except FileNotFoundError:
    st.error("Models/data not found. Run the pipeline first."); st.stop()

# ── Per-country keywords ────────────────────────────────────────────────────
st.subheader("Top Keywords by Country")

top_n = st.sidebar.slider("Keywords to show", 5, 30, 15, key="tfidf_top_n")
country_kw = keywords_by_country(df, vectorizer, top_n=top_n)

tabs = st.tabs(list(country_kw.keys()))
for tab, (country, kws) in zip(tabs, country_kw.items()):
    with tab:
        if kws:
            kw_df = pd.DataFrame(kws)
            fig = px.bar(
                kw_df, x="score", y="keyword", orientation="h",
                title=f"{country} — Top {top_n} Keywords",
                color="score", color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No keywords for {country}")

# ── Burst-period keywords ──────────────────────────────────────────────────
st.divider()
st.subheader("Keywords During Burst Periods")

burst_kw = keywords_for_bursts(df, burst_df, vectorizer, top_n=top_n)

if burst_kw:
    tabs2 = st.tabs(list(burst_kw.keys()))
    for tab, (country, kws) in zip(tabs2, burst_kw.items()):
        with tab:
            if kws:
                kw_df = pd.DataFrame(kws)
                fig = px.bar(
                    kw_df, x="score", y="keyword", orientation="h",
                    title=f"{country} — Burst Period Keywords",
                    color="score", color_continuous_scale="Magma",
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No burst keywords for {country}")
else:
    st.info("No burst periods detected — nothing to show here.")
