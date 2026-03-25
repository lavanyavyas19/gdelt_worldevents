"""
Page 3 — Event Types
Conflict vs cooperation distribution, bar charts per country, ratio comparison.
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


st.header("Event Types")

try:
    df = load_events()
except FileNotFoundError:
    st.error("Processed data not found."); st.stop()

countries = st.sidebar.multiselect(
    "Countries", df["country"].unique().tolist(),
    default=df["country"].unique().tolist(), key="et_countries",
)
df = df[df["country"].isin(countries)]

# ── Overall QuadClass distribution ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    quad_counts = df["QuadLabel"].value_counts().reset_index()
    quad_counts.columns = ["QuadLabel", "count"]
    fig1 = px.pie(quad_counts, names="QuadLabel", values="count",
                  title="Event Classification (All Countries)",
                  color="QuadLabel",
                  color_discrete_map={
                      "Verbal Cooperation": "#00CC96",
                      "Material Cooperation": "#19D3F3",
                      "Verbal Conflict": "#FFA15A",
                      "Material Conflict": "#EF553B",
                  })
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    simple = df.groupby(["country", "EventType"]).size().reset_index(name="count")
    fig2 = px.bar(simple, x="country", y="count", color="EventType",
                  barmode="group", title="Conflict vs Cooperation by Country",
                  color_discrete_map={"Conflict": "#EF553B", "Cooperation": "#00CC96"})
    st.plotly_chart(fig2, use_container_width=True)

# ── Detailed QuadClass per country ──────────────────────────────────────────
detail = df.groupby(["country", "QuadLabel"]).size().reset_index(name="count")
fig3 = px.bar(detail, x="QuadLabel", y="count", color="country",
              barmode="group", title="Detailed Event Classification by Country")
st.plotly_chart(fig3, use_container_width=True)

# ── Ratio comparison ────────────────────────────────────────────────────────
st.subheader("Conflict / Cooperation Ratio")
ratio_df = df.groupby("country").apply(
    lambda g: pd.Series({
        "conflict_ratio": (g["EventType"] == "Conflict").mean(),
        "cooperation_ratio": (g["EventType"] == "Cooperation").mean(),
    })
).reset_index()

fig4 = px.bar(
    ratio_df.melt(id_vars="country", var_name="metric", value_name="ratio"),
    x="country", y="ratio", color="metric", barmode="group",
    title="Conflict & Cooperation Ratios",
    color_discrete_map={"conflict_ratio": "#EF553B", "cooperation_ratio": "#00CC96"},
)
st.plotly_chart(fig4, use_container_width=True)
