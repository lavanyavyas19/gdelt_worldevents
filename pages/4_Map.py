"""
Page 4 — Map
Geographic hotspot visualization using ActionGeo coordinates.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.storage import load_df
PROCESSED = os.path.join(ROOT, "data", "processed")


@st.cache_data
def load_events():
    return load_df(os.path.join(PROCESSED, "events"))


st.header("Geographic Hotspots")

try:
    df = load_events()
except FileNotFoundError:
    st.error("Processed data not found."); st.stop()

countries = st.sidebar.multiselect(
    "Countries", df["country"].unique().tolist(),
    default=df["country"].unique().tolist(), key="map_countries",
)
event_type = st.sidebar.multiselect(
    "Event Type", ["Conflict", "Cooperation"],
    default=["Conflict", "Cooperation"], key="map_etype",
)

df = df[df["country"].isin(countries) & df["EventType"].isin(event_type)]

# Drop rows without coordinates
geo_df = df.dropna(subset=["ActionGeo_Lat", "ActionGeo_Long"]).copy()

# Sample for performance (maps with >50k points are slow)
MAX_MAP_POINTS = 20_000
if len(geo_df) > MAX_MAP_POINTS:
    geo_df = geo_df.sample(MAX_MAP_POINTS, random_state=42)
    st.info(f"Showing a random sample of {MAX_MAP_POINTS:,} events for performance.")

# ── Scatter-mapbox ──────────────────────────────────────────────────────────
fig = px.scatter_mapbox(
    geo_df,
    lat="ActionGeo_Lat",
    lon="ActionGeo_Long",
    color="EventType",
    hover_name="ActionGeo_FullName",
    hover_data=["country", "QuadLabel", "AvgTone"],
    color_discrete_map={"Conflict": "#EF553B", "Cooperation": "#00CC96"},
    zoom=1,
    height=650,
    title="Event Locations",
)
fig.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig, use_container_width=True)

# ── Top locations table ─────────────────────────────────────────────────────
st.subheader("Top Event Locations")
top_locs = (
    df.groupby(["ActionGeo_FullName", "country"])
    .agg(events=("GLOBALEVENTID", "count"), avg_tone=("AvgTone", "mean"))
    .reset_index()
    .nlargest(20, "events")
)
st.dataframe(top_locs, use_container_width=True)
