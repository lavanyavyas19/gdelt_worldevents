"""
Page 7 — Event Chain Explorer
One question: What happened before and after this event?
"""

import streamlit as st
import pandas as pd
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, show_data_window, data_not_found, empty_state,
    tone_label, tone_with_value, match_strength, format_reasons,
)
from src.chains import find_chain

try:
    df = load_events()
except FileNotFoundError:
    data_not_found()

show_data_window()

# ── Header ────────────────────────────────────────────────────────────────────
st.header("Event Chain Explorer")
st.caption("Pick an event to see what happened before and after it.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
country_filter = st.sidebar.selectbox(
    "Country", sorted(df["country"].dropna().unique().tolist()), key="chain_country",
)
window = st.sidebar.slider("Time window (days)", 1, 21, 7, key="chain_window")
top_n = st.sidebar.slider("Related events to show", 1, 15, 5, key="chain_topn")

event_type_filter = st.sidebar.multiselect(
    "Event type (optional)", ["Conflict", "Cooperation"],
    default=[], key="chain_etype",
)

country_df = df[df["country"] == country_filter].copy()
if country_df.empty:
    empty_state(f"No events found for {country_filter}.")
    st.stop()

if event_type_filter:
    picker_df = country_df[country_df["EventType"].isin(event_type_filter)]
else:
    picker_df = country_df

if picker_df.empty:
    empty_state("No events match current filters.")
    st.stop()

# ── Event picker ──────────────────────────────────────────────────────────────
st.subheader("Start with one event")

sample_size = min(200, len(picker_df))
sample = picker_df.sample(sample_size, random_state=42).sort_values("event_date")

# Build clean picker labels: "Mar 23 — Actor → Actor — Verbal Conflict"
def _picker_label(r):
    date = r["event_date"].strftime("%b %d")
    a1 = r["actor1_clean"] if r["actor1_clean"] != "Unknown" else "—"
    a2 = r["actor2_clean"] if r["actor2_clean"] != "Unknown" else "—"
    quad = r.get("QuadLabel", "")
    eid = int(r["GLOBALEVENTID"])
    return f"{date}  ·  {a1} → {a2}  ·  {quad}  ({eid})"

event_options = sample.apply(_picker_label, axis=1).tolist()
event_ids = sample["GLOBALEVENTID"].astype(int).tolist()

selected_idx = st.selectbox(
    "Browse events:", range(len(event_options)),
    format_func=lambda i: event_options[i],
    key="chain_event_select",
)
event_id = event_ids[selected_idx]

# ── Build chain ───────────────────────────────────────────────────────────────
with st.spinner("Finding related events…"):
    chain = find_chain(df, event_id, window_days=window, top_n=top_n)

# ── Selected event card ───────────────────────────────────────────────────────
st.divider()
st.subheader("Selected Event")

if chain["selected"]:
    sel = chain["selected"]
    date_str = str(sel.get("event_date", ""))[:10]
    a1 = sel.get("actor1_clean", "Unknown")
    a2 = sel.get("actor2_clean", "Unknown")
    actors_str = f"{a1} → {a2}" if a1 != "Unknown" or a2 != "Unknown" else "Actors not identified"
    location = sel.get("ActionGeo_FullName", "Unknown")
    quad = sel.get("QuadLabel", "Unknown")
    root = sel.get("EventRootLabel", "")
    tone_val = sel.get("AvgTone", 0)
    url = sel.get("SOURCEURL", "")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**{date_str}** · {location}")
        st.markdown(actors_str)
    with col2:
        st.markdown(f"**{quad}**" + (f" · {root}" if root else ""))
        st.markdown(f"Tone: {tone_with_value(tone_val)}")
    with col3:
        if url and isinstance(url, str) and url.startswith("http"):
            st.link_button("Read source", url)

# ── Auto-generated chain narrative ────────────────────────────────────────────
n_prev = len(chain.get("previous", []))
n_next = len(chain.get("next", []))

if chain["selected"]:
    quad_lower = quad.lower() if quad != "Unknown" else "event"
    root_lower = root.lower() if root else ""

    parts = []
    if n_prev > 0 or n_next > 0:
        parts.append(
            f"This {quad_lower} event"
            + (f" ({root_lower})" if root_lower else "")
            + f" in {country_filter} is part of a sequence of"
            f" {n_prev + n_next} related events over ±{window} days."
        )
    if n_prev == 0:
        parts.append("No strong preceding events were found.")
    if n_next == 0:
        parts.append("No strong follow-up events were found.")

    st.info(" ".join(parts))

st.divider()


# ── Card renderer ─────────────────────────────────────────────────────────────
def render_event_card(ev: dict, max_score: float):
    """Render a single related event as a clean card."""
    score = ev.get("chain_score", 0)
    raw_reasons = ev.get("score_reasons", "")
    reasons = format_reasons(raw_reasons)
    date_str = str(ev.get("event_date", ""))[:10]
    a1 = ev.get("actor1_clean", "Unknown")
    a2 = ev.get("actor2_clean", "Unknown")
    actors = f"{a1} → {a2}" if a1 != "Unknown" or a2 != "Unknown" else "Actors not identified"
    quad = ev.get("QuadLabel", "Unknown")
    tone_val = ev.get("AvgTone", 0)
    location = ev.get("ActionGeo_FullName", "Unknown")
    source = ev.get("SOURCEURL", "")
    strength = match_strength(score, max_score)

    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{date_str}** · {location}")
            st.markdown(f"{actors} · {quad} · Tone: {tone_with_value(tone_val)}")
            if reasons:
                pills = "  ·  ".join(reasons)
                st.caption(f"Why matched: {pills}")
        with cols[1]:
            st.metric("Match", f"{strength}", f"{score:.0f} pts")
            if source and isinstance(source, str) and source.startswith("http"):
                st.markdown(f"[Source]({source})")
        st.markdown("---")


# ── Previous events ───────────────────────────────────────────────────────────
st.subheader("What Happened Before")

if chain["previous"]:
    all_scores = [e.get("chain_score", 0) for e in chain["previous"] + chain["next"]]
    max_s = max(all_scores) if all_scores else 1
    for ev in chain["previous"]:
        render_event_card(ev, max_s)
else:
    st.caption("No related events found before this one.")

# ── Next events ───────────────────────────────────────────────────────────────
st.subheader("What Happened After")

if chain["next"]:
    all_scores = [e.get("chain_score", 0) for e in chain["previous"] + chain["next"]]
    max_s = max(all_scores) if all_scores else 1
    for ev in chain["next"]:
        render_event_card(ev, max_s)
else:
    st.caption("No strong related events were found after this event.")

# ── Technical details (hidden) ────────────────────────────────────────────────
with st.expander("Show technical details"):
    total_linked = n_prev + n_next
    all_scores = [e.get("chain_score", 0) for e in chain.get("previous", []) + chain.get("next", [])]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    st.markdown(f"""
**Chain summary**

- Related events found: {total_linked}
- Time window: ±{window} days
- Average relevance score: {avg_score:.1f}

**Scoring method**

Each candidate event is scored on: same country (+3), shared actor (+3),
same event family (+2), same conflict/cooperation class (+2), same location (+1),
similar tone (+1), and time proximity (up to +1).
    """)
