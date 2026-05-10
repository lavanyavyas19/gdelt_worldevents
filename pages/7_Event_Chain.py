import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, show_data_window, data_not_found, empty_state,
    tone_label, tone_with_value, match_strength, format_reasons,
    goldstein_label, PATTERN_COLORS,
)
from src.chains import find_chain
from src.config import CHAIN_MAX_POSSIBLE

try:
    df = load_events()
except FileNotFoundError:
    data_not_found()

show_data_window()

st.header("Event Chain Explorer")
st.caption("Trace what happened before and after any event in the dataset.")


_spike_date    = st.session_state.get("selected_date")   
_spike_country = st.session_state.get("selected_country") 
if _spike_date:
    banner_col, clear_col = st.columns([8, 1])
    with banner_col:
        st.info(
            f"Investigating spike period: **{_spike_date}**"
            + (f" · {_spike_country}" if _spike_country else "")
            + "  —  showing events ±7 days around this date."
        )
    with clear_col:
        st.write("")
        if st.button("Clear", key="clear_spike_ctx"):
            st.session_state.pop("selected_date", None)
            st.session_state.pop("selected_country", None)
            st.rerun()


params = st.query_params
entry_country  = params.get("chain_country", None)
entry_date     = params.get("chain_date", None)
entry_event_id = params.get("chain_event", None)


all_countries = sorted(df["country"].dropna().unique().tolist())

_default_country = _spike_country or entry_country
default_idx = all_countries.index(_default_country) if _default_country in all_countries else 0

country_filter = st.sidebar.selectbox(
    "Country", all_countries, index=default_idx, key="chain_country",
)
window = st.sidebar.slider("Time window (days)", 1, 21, 7, key="chain_window")
top_n  = st.sidebar.slider("Related events to show", 1, 10, 5, key="chain_topn")
event_type_filter = st.sidebar.multiselect(
    "Event type (optional)", ["Conflict", "Cooperation"],
    default=[], key="chain_etype",
)


country_df = df[df["country"] == country_filter].copy()
if country_df.empty:
    empty_state(f"No events found for {country_filter}.")
    st.stop()

picker_df = (
    country_df[country_df["EventType"].isin(event_type_filter)]
    if event_type_filter else country_df
)
if picker_df.empty:
    empty_state("No events match current filters.")
    st.stop()


if _spike_date:
    try:
        spike_ts = pd.to_datetime(_spike_date)
        date_window = picker_df[
            (picker_df["event_date"] >= spike_ts - pd.Timedelta(days=7))
            & (picker_df["event_date"] <= spike_ts + pd.Timedelta(days=7))
        ]
        if not date_window.empty:
            picker_df = date_window
    except (ValueError, TypeError):
        pass  

st.subheader("Select an Event")
search_col, hint_col = st.columns([3, 1])
with search_col:
    search_text = st.text_input(
        "Search by actor, location, or event ID",
        placeholder="e.g., Military, Tehran, 12345",
        key="chain_search",
    )
with hint_col:
    if entry_date:
        st.caption(f"Linked from: {entry_date}")


if search_text:
    q = search_text.strip().lower()
    try:
        sid = int(search_text.strip())
        mask = picker_df["GLOBALEVENTID"] == sid
    except ValueError:
        mask = (
            picker_df["actor1_clean"].str.lower().str.contains(q, na=False)
            | picker_df["actor2_clean"].str.lower().str.contains(q, na=False)
            | picker_df["ActionGeo_FullName"].fillna("").str.lower().str.contains(q, na=False)
        )
    results = picker_df[mask]
    if results.empty:
        st.warning(f"No matches for '{search_text}'. Showing top events instead.")
        results = picker_df
else:
    results = picker_df


if entry_date:
    try:
        target = pd.to_datetime(entry_date)
        date_match = results[results["day"] == target]
        if not date_match.empty:
            results = date_match
    except (ValueError, TypeError):
        pass

if entry_event_id:
    try:
        tid = int(entry_event_id)
        id_match = results[results["GLOBALEVENTID"] == tid]
        if not id_match.empty:
            results = id_match
    except (ValueError, TypeError):
        pass

display_events = (
    results
    .sort_values("event_strength", ascending=False)
    .head(100)
    .sort_values("event_date", ascending=False)
)

def _fmt_date(v) -> str:
    """Normalize any date representation to YYYY-MM-DD."""
    if pd.isna(v) if not isinstance(v, str) else (v == ""):
        return ""
    return str(v)[:10]

def _picker_label(r) -> str:
    date = r["event_date"].strftime("%b %d")
    a1 = r["actor1_clean"] if r["actor1_clean"] != "Unknown" else "—"
    a2 = r["actor2_clean"] if r["actor2_clean"] != "Unknown" else "—"
    return f"{date}  ·  {a1} → {a2}  ·  {r.get('QuadLabel', '')}"

event_options = display_events.apply(_picker_label, axis=1).tolist()
event_ids     = display_events["GLOBALEVENTID"].astype(int).tolist()

if not event_options:
    empty_state("No events to display.")
    st.stop()

selected_idx = st.selectbox(
    f"Choose from {len(event_options)} events:",
    range(len(event_options)),
    format_func=lambda i: event_options[i],
    key="chain_event_select",
)
event_id = event_ids[selected_idx]


with st.spinner("Building event chain…"):
    chain = find_chain(df, event_id, window_days=window, top_n=top_n)

if not chain["selected"]:
    empty_state("Could not build a chain for this event.")
    st.stop()


st.divider()
pattern       = chain.get("pattern", "Unknown")
pattern_color = PATTERN_COLORS.get(pattern, "#999999")
n_prev        = len(chain.get("previous", []))
n_next        = len(chain.get("next", []))

col_pat, col_stats = st.columns([3, 1])
with col_pat:
    st.markdown(
        f"**Chain Pattern:** "
        f"<span style='background:{pattern_color};color:white;"
        f"padding:3px 12px;border-radius:12px;font-size:0.85em;'>"
        f"{pattern}</span>",
        unsafe_allow_html=True,
    )
with col_stats:
    st.caption(f"{n_prev} preceding  ·  {n_next} following  ·  ±{window}d window")

narrative = chain.get("narrative", "")
if narrative:
    st.info(narrative)


def _build_timeline(events: list, anchor: dict, color: str):
    """Compact Plotly dot-timeline across all chain events."""
    dates, labels, colors, sizes, hovers = [], [], [], [], []
    anchor_id = anchor.get("GLOBALEVENTID", "")

    for ev in events:
        raw = ev.get("event_date", "")
        try:
            d = pd.to_datetime(str(raw)[:10])
        except Exception:
            continue
        if pd.isna(d):
            continue

        etype = ev.get("EventType", "Unknown")
        a1    = ev.get("actor1_clean", "?")
        a2    = ev.get("actor2_clean", "?")
        quad  = ev.get("QuadLabel", "?")
        score = ev.get("chain_score", "")
        is_anchor = ev.get("GLOBALEVENTID", "") == anchor_id

        dates.append(d)
        labels.append("ANCHOR" if is_anchor else quad[:18])
        colors.append(color if is_anchor else ("#EF553B" if etype == "Conflict" else "#00CC96"))
        sizes.append(16 if is_anchor else 9)
        hovers.append(
            f"{_fmt_date(raw)}<br>{a1} → {a2}<br>{quad}"
            + (f"<br>Score: {score:.2f}" if isinstance(score, float) else "")
        )

    if not dates:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=[0] * len(dates),
        mode="lines",
        line=dict(color="#DDDDDD", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[0] * len(dates),
        mode="markers+text",
        marker=dict(color=colors, size=sizes, line=dict(width=1, color="white")),
        text=labels,
        textposition="top center",
        textfont=dict(size=8),
        hovertext=hovers,
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=8, b=28),
        yaxis=dict(visible=False, range=[-0.5, 0.8]),
        xaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


all_events = chain["previous"] + [chain["selected"]] + chain["next"]
if len(all_events) > 1:
    _build_timeline(all_events, chain["selected"], pattern_color)

st.divider()


_STRENGTH_COLOR = {"Strong": "#00CC96", "Moderate": "#FFA15A", "Weak": "#AAAAAA"}


def _render_anchor_card(ev: dict):
    """Large highlighted card for the anchor event."""
    date_str  = _fmt_date(ev.get("event_date", ""))
    a1        = ev.get("actor1_clean", "Unknown")
    a2        = ev.get("actor2_clean", "Unknown")
    actors    = f"{a1} &nbsp;→&nbsp; {a2}" if a1 != "Unknown" or a2 != "Unknown" else "Actors not identified"
    quad      = ev.get("QuadLabel", "Unknown")
    root      = ev.get("EventRootLabel", "")
    etype     = ev.get("EventType", "Unknown")
    tone_val  = ev.get("AvgTone", 0)
    gold_val  = ev.get("GoldsteinScale", "")
    location  = ev.get("ActionGeo_FullName", "") or "Location unknown"
    url       = ev.get("SOURCEURL", "")

    border   = "#EF553B" if etype == "Conflict" else "#00CC96"
    bg       = "rgba(239,85,59,0.05)" if etype == "Conflict" else "rgba(0,204,150,0.05)"
    tone_str = tone_with_value(tone_val)
    gold_str = f"&nbsp;·&nbsp; {goldstein_label(gold_val)}" if gold_val != "" else ""
    src_html = (
        f"&nbsp;·&nbsp; <a href='{url}' target='_blank' "
        f"style='font-size:0.82em;color:#666;'>Read source</a>"
        if url and str(url).startswith("http") else ""
    )
    sub_line = f"<b>{quad}</b>" + (f" &nbsp;·&nbsp; {root}" if root else "")

    st.markdown(f"""
<div style="border:2px solid {border}; background:{bg};
            border-radius:8px; padding:16px 18px; margin:4px 0 12px 0;">
  <div style="font-size:0.8em; color:#888; margin-bottom:4px;">
    {date_str} &nbsp;·&nbsp; {location}{src_html}
  </div>
  <div style="font-size:1.05em; font-weight:700; margin-bottom:4px;">{actors}</div>
  <div style="font-size:0.9em; color:#444; margin-bottom:4px;">{sub_line}</div>
  <div style="font-size:0.82em; color:#666;">Tone: {tone_str} &nbsp;{gold_str}</div>
</div>
""", unsafe_allow_html=True)


def _render_event_card(ev: dict, max_score: float):
    """Compact related-event card with relevance score and reason pills."""
    score     = ev.get("chain_score", 0)
    reasons   = format_reasons(ev.get("score_reasons", ""))
    date_str  = _fmt_date(ev.get("event_date", ""))
    a1        = ev.get("actor1_clean", "Unknown")
    a2        = ev.get("actor2_clean", "Unknown")
    actors    = f"{a1} &nbsp;→&nbsp; {a2}" if a1 != "Unknown" or a2 != "Unknown" else "Actors not identified"
    quad      = ev.get("QuadLabel", "Unknown")
    etype     = ev.get("EventType", "Unknown")
    tone_val  = ev.get("AvgTone", 0)
    gold_val  = ev.get("GoldsteinScale", "")
    location  = ev.get("ActionGeo_FullName", "") or "Location unknown"
    url       = ev.get("SOURCEURL", "")
    strength  = match_strength(score, max_score)

    border  = "#EF553B" if etype == "Conflict" else "#00CC96"
    bg      = "rgba(239,85,59,0.04)" if etype == "Conflict" else "rgba(0,204,150,0.04)"
    s_color = _STRENGTH_COLOR.get(strength, "#AAAAAA")

    tone_str = tone_with_value(tone_val)
    gold_str = f"&nbsp;·&nbsp; {goldstein_label(gold_val)}" if gold_val != "" else ""
    src_html = (
        f"&nbsp;·&nbsp; <a href='{url}' target='_blank' "
        f"style='font-size:0.8em;color:#888;'>Source</a>"
        if url and str(url).startswith("http") else ""
    )
    pills_html = ""
    if reasons:
        pills = " ".join(
            f"<span style='background:#f0f2f6;padding:1px 7px;"
            f"border-radius:8px;font-size:0.76em;'>{r}</span>"
            for r in reasons
        )
        pills_html = f"<div style='margin-top:5px;'>{pills}</div>"

    st.markdown(f"""
<div style="border-left:4px solid {border}; background:{bg};
            padding:11px 14px; border-radius:0 6px 6px 0; margin-bottom:8px;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <div style="flex:1; min-width:0;">
      <div style="font-size:0.78em; color:#888; margin-bottom:2px;">
        {date_str} &nbsp;·&nbsp; {location}{src_html}
      </div>
      <div style="font-size:0.92em; font-weight:600; margin-bottom:2px;">{actors}</div>
      <div style="font-size:0.83em; color:#555; margin-bottom:3px;">{quad}</div>
      <div style="font-size:0.79em; color:#777;">Tone: {tone_str}{gold_str}</div>
      {pills_html}
    </div>
    <div style="text-align:right; margin-left:14px; min-width:56px; flex-shrink:0;">
      <div style="font-size:1.05em; font-weight:700; color:{s_color};">{strength}</div>
      <div style="font-size:0.73em; color:#aaa;">{score:.1f} pts</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)



all_chain = chain.get("previous", []) + chain.get("next", [])
max_s = max((e.get("chain_score", 0) for e in all_chain), default=1.0)


st.subheader("Previous Events")
if chain["previous"]:
    for ev in chain["previous"]:
        _render_event_card(ev, max_s)
else:
    st.caption("No related events found in the preceding window.")


st.markdown(
    "<div style='text-align:center; font-size:1.4em; color:#BBBBBB; "
    "margin:6px 0;'>↓</div>",
    unsafe_allow_html=True,
)


st.subheader("Anchor Event")
_render_anchor_card(chain["selected"])


st.markdown(
    "<div style='text-align:center; font-size:1.4em; color:#BBBBBB; "
    "margin:6px 0;'>↓</div>",
    unsafe_allow_html=True,
)


st.subheader("Next Events")
if chain["next"]:
    for ev in chain["next"]:
        _render_event_card(ev, max_s)
else:
    st.caption("No strongly related events found in the following window.")


st.markdown("")
with st.expander("Scoring details"):
    total_linked = n_prev + n_next
    all_scores   = [e.get("chain_score", 0) for e in all_chain]
    avg_score    = sum(all_scores) / len(all_scores) if all_scores else 0

    st.markdown(f"""
**Chain summary**

- Related events: {total_linked}  ·  Pattern: {pattern}  ·  Window: ±{window} days
- Average relevance score: {avg_score:.1f} / {CHAIN_MAX_POSSIBLE}

**Scoring method**

Events are ranked with a 16-feature vector covering: same country, shared actor
(exact and fuzzy), event family, conflict/cooperation class, location, tone distance,
Goldstein intensity distance, temporal decay (τ = 3 days), event importance, and
cross-country links. Scored by learned logistic regression weights or heuristic
fallback when no model is trained.

**Pattern classification**

The chain pattern (Escalation, De-escalation, Persistence, Mixed, Isolated) is derived
from a Mann-Kendall trend test on the Goldstein scale trajectory across the chain.
    """)
