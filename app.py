"""
app.py — GDELT Event Intelligence Dashboard
=============================================
Landing page for the Streamlit multi-page application.

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
    [data-testid="stMetricValue"]         { font-size: 1.3rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    .block-container                       { padding-top: 2rem; }
    .feature-card {
        border: 1px solid #e4e4e4;
        border-radius: 10px;
        padding: 20px 22px;
        background: #fafafa;
        height: 100%;
    }
    .feature-card h4 { margin: 0 0 6px 0; font-size: 1.0em; font-weight: 700; }
    .feature-card p  { margin: 0; font-size: 0.85em; color: #555; line-height: 1.5; }
    .feature-tag {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 6px;
        padding: 2px 9px;
        font-size: 0.75em;
        color: #444;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.title("Global Event Intelligence")
st.markdown(
    "**USA · India · Iran** &nbsp;—&nbsp; December 2025 – March 2026",
    unsafe_allow_html=True,
)
st.markdown(
    "An intelligence analysis system for tracking conflict, cooperation, and "
    "geopolitical shifts using structured event data from the "
    "[GDELT Project](https://www.gdeltproject.org/)."
)

st.markdown("")
st.divider()

# ── Feature cards ──────────────────────────────────────────────────────────────
st.subheader("Explore the System")
st.caption("Each view answers a distinct intelligence question.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">Summary</span>
  <h4>Overview</h4>
  <p>Key metrics, per-country activity shifts, conflict/cooperation balance,
     and auto-generated intelligence insights.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/1_Overview.py", label="Open Overview")

with c2:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">Time Series</span>
  <h4>Trends</h4>
  <p>Weekly event volume and tone trajectories across all countries.
     Spike weeks highlighted inline.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/2_Trends.py", label="Open Trends")

with c3:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">Anomaly Detection</span>
  <h4>Activity Spikes</h4>
  <p>Statistical detection of unusual event surges. Drill into any spike
     to investigate the events that drove it.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/6_Burst_Dashboard.py", label="Open Activity Spikes")

with c4:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">Causal Chain</span>
  <h4>Event Chain</h4>
  <p>Given any event, surface what happened before and after it —
     scored by actor overlap, temporal proximity, and intensity.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/7_Event_Chain.py", label="Open Event Chain")

st.markdown("")

# Second row
c5, c6, c7, _ = st.columns(4)

with c5:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">NLP</span>
  <h4>Characteristic Terms</h4>
  <p>TF-IDF keyword patterns by country — and how they shift
     during activity spikes vs. normal periods.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/8_Topic_Lens.py", label="Open Topic Lens")

with c6:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">Geospatial</span>
  <h4>Event Map</h4>
  <p>Geographic distribution of events. Filter by country, type,
     and time period to locate hotspots.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/4_Map.py", label="Open Map")

with c7:
    st.markdown("""
<div class="feature-card">
  <span class="feature-tag">Event Types</span>
  <h4>Event Breakdown</h4>
  <p>Distribution across CAMEO event families —
     from verbal cooperation to material conflict.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/3_Event_Types.py", label="Open Event Types")

st.divider()

# ── Workflow callout ───────────────────────────────────────────────────────────
st.markdown("""
**Recommended workflow**

1. Start at **Overview** to understand the big picture across all three countries.
2. Go to **Activity Spikes** to find the most unusual days.
3. Click **Investigate** on any spike to jump into **Event Chain** and see
   exactly what drove it.
4. Use **Topic Lens** to understand which actor and location patterns dominate
   during high-activity periods.
""")

st.caption(
    "First run? Execute `python -m src.prepare_data` to build the dataset.  "
    "Data source: [GDELT Project v1](https://www.gdeltproject.org/)"
)
