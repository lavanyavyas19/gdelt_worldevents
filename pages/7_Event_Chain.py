"""
Page 7 — Event Chain Explorer
Select an event and see related previous/next events ranked by relevance.
"""

import streamlit as st
import pandas as pd
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.storage import load_df

from src.build_chain import find_chain

PROCESSED = os.path.join(ROOT, "data", "processed")


@st.cache_data
def load_events():
    return load_df(os.path.join(PROCESSED, "events"))


st.header("Event Chain Explorer")

try:
    df = load_events()
except FileNotFoundError:
    st.error("Processed data not found."); st.stop()

# ── Sidebar controls ────────────────────────────────────────────────────────
country_filter = st.sidebar.selectbox("Country", df["country"].unique().tolist())
window = st.sidebar.slider("Time Window (days)", 1, 14, 5)
top_n = st.sidebar.slider("Chain Size", 1, 10, 5)

# Filter to selected country for the event picker
country_df = df[df["country"] == country_filter]

# Let user pick by browsing a sample
sample = country_df.sample(min(100, len(country_df)), random_state=42).sort_values("SQLDATE")
event_options = sample.apply(
    lambda r: f"{int(r['GLOBALEVENTID'])} | {r['SQLDATE'].date()} | {r['Actor1Name']} → {r['Actor2Name']} | {r['QuadLabel']}",
    axis=1,
).tolist()

selected = st.selectbox("Select an event", event_options)
event_id = int(selected.split("|")[0].strip())

# ── Build chain ─────────────────────────────────────────────────────────────
chain = find_chain(df, event_id, window_days=window, top_n=top_n)

# ── Display ─────────────────────────────────────────────────────────────────
st.subheader("Anchor Event")
if chain["selected"]:
    sel = chain["selected"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event ID", int(sel.get("GLOBALEVENTID", 0)))
    c2.metric("Date", str(sel.get("SQLDATE", ""))[:10])
    c3.metric("Type", sel.get("QuadLabel", ""))
    c4.metric("Tone", f"{sel.get('AvgTone', 0):.2f}")

    st.markdown(f"**Actors:** {sel.get('Actor1Name', 'Unknown')} → {sel.get('Actor2Name', 'Unknown')}")
    if sel.get("SOURCEURL"):
        st.markdown(f"[Source Article]({sel['SOURCEURL']})")

st.divider()

# Previous events
st.subheader(f"Previous Events (before anchor)")
if chain["previous"]:
    prev_df = pd.DataFrame(chain["previous"])
    st.dataframe(
        prev_df,
        use_container_width=True,
        column_config={"SOURCEURL": st.column_config.LinkColumn("Source")},
    )
else:
    st.info("No related previous events found in the time window.")

# Next events
st.subheader(f"Next Events (after anchor)")
if chain["next"]:
    next_df = pd.DataFrame(chain["next"])
    st.dataframe(
        next_df,
        use_container_width=True,
        column_config={"SOURCEURL": st.column_config.LinkColumn("Source")},
    )
else:
    st.info("No related next events found in the time window.")

# ── Chain summary ───────────────────────────────────────────────────────────
st.divider()
total_linked = len(chain.get("previous", [])) + len(chain.get("next", []))
st.caption(f"Chain retrieved {total_linked} related events within ±{window} days.")
