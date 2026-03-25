"""
app.py — GDELT Event Intelligence Dashboard
=============================================
Main entry point for the Streamlit multi-page application.

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="GDELT Event Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GDELT Event Intelligence Dashboard")
st.markdown(
    """
    Welcome to the **GDELT Event Intelligence System** — a hybrid analytics dashboard
    that combines country-level event analysis with smart event-chain exploration,
    burst detection, and keyword intelligence.

    **Countries analysed:** USA · India · Iran

    ---

    ### How to navigate

    Use the **sidebar** to open any page:

    | Page | What it shows |
    |------|---------------|
    | **Overview** | KPI cards, total events, average tone, burst count |
    | **Trends** | Event count & sentiment over time |
    | **Event Types** | Conflict vs Cooperation breakdown |
    | **Map** | Geographic hotspot visualization |
    | **Dataset Explorer** | Filterable event table with source links |
    | **Burst Dashboard** | Spike detection and analysis |
    | **Event Chain** | Related event retrieval (before → anchor → after) |
    | **Topic Lens** | TF-IDF keyword extraction per country & burst period |

    ---

    ### Data source

    [GDELT Project](https://www.gdeltproject.org/) — Global Database of Events, Language, and Tone.

    ---

    > **First run?**  Execute the data pipeline first:
    > ```bash
    > python -m src.prepare_data
    > ```
    """
)
