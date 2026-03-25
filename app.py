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

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("Global Event Intelligence")
st.caption("USA · India · Iran  —  December – March window")

st.markdown("""
Explore how conflict, cooperation, and public discourse shift across
three countries using event-level data from the GDELT project.

Use the **sidebar** to navigate between views.
""")

st.markdown("")

col1, col2, col3, col4 = st.columns(4)
col1.markdown("**Overview** — Key numbers at a glance")
col2.markdown("**Trends** — How activity changes over time")
col3.markdown("**Burst Detection** — Unusual spikes in events")
col4.markdown("**Event Chains** — Related events before and after")

st.markdown("")

st.caption(
    "Data source: [GDELT Project](https://www.gdeltproject.org/)  ·  "
    "First run? Execute `python -m src.prepare_data` to build the dataset."
)
