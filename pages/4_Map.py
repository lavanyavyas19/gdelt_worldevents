import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, sidebar_country_filter, sidebar_event_type_filter,
    show_data_window, apply_filters, data_not_found, empty_state,
)
from src.config import COLOR_MAP_EVENT, MAX_MAP_POINTS

try:
    df = load_events()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(df, key="map_countries")
event_types = sidebar_event_type_filter(key="map_etype")

df = apply_filters(df, countries, event_types)

st.header("Where Events Happen")

if df.empty:
    empty_state()
    st.stop()

geo_df = df.dropna(subset=["ActionGeo_Lat", "ActionGeo_Long"]).copy()

if geo_df.empty:
    empty_state("No events with location data match your filters.")
    st.stop()

sampled = False
if len(geo_df) > MAX_MAP_POINTS:
    geo_df = geo_df.sample(MAX_MAP_POINTS, random_state=42)
    sampled = True

if sampled:
    st.caption(f"Showing {MAX_MAP_POINTS:,} sampled events for performance.")

# ── Map (primary focus) ───────────────────────────────────────────────────────
fig = px.scatter_mapbox(
    geo_df,
    lat="ActionGeo_Lat",
    lon="ActionGeo_Long",
    color="EventType",
    hover_name="ActionGeo_FullName",
    hover_data={"country": True, "QuadLabel": True, "AvgTone": ":.1f",
                "ActionGeo_Lat": False, "ActionGeo_Long": False},
    color_discrete_map=COLOR_MAP_EVENT,
    zoom=1.5,
    height=620,
    labels={"EventType": "Type", "country": "Country",
            "QuadLabel": "Event Class", "AvgTone": "Tone"},
)
fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

# ── Top 5 locations ───────────────────────────────────────────────────────────
st.markdown("")
st.subheader("Busiest Locations")

top_locs = (
    df.groupby(["ActionGeo_FullName", "country"])
    .agg(Events=("GLOBALEVENTID", "count"), Tone=("AvgTone", "mean"))
    .reset_index()
    .rename(columns={"ActionGeo_FullName": "Location", "country": "Country"})
    .nlargest(5, "Events")
)
top_locs["Tone"] = top_locs["Tone"].round(1)
st.dataframe(top_locs, use_container_width=True, hide_index=True)
