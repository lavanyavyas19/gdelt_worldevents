"""
Page 6 — Activity Spikes + Intelligence Hub
One question: When did something unusual happen — and what does it mean?

Tabs:
  1. Summary          — cross-country spike overview
  2. Country Analysis — per-country z-score timeline
  3. Spike Details    — table + drill-down to Event Chain
  4. Intelligence Hub — AI summarisation, RAG evidence, Q&A, comparison, PDF
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (
    load_events, load_bursts, sidebar_country_filter,
    show_data_window, data_not_found, empty_state, metric_row,
)
from src.config import COLOR_MAP_COUNTRY, BURST_ROLLING_WINDOW, BURST_Z_THRESHOLD
from src.burst import detect_bursts, get_burst_summary

# ── Intelligence module imports (all optional — degrade gracefully) ───────────
try:
    from src.summarizer import (
        summarize_spike, spike_info_from_row, get_event_type_dist,
        is_ollama_available,
    )
    _HAS_SUMMARIZER = True
except ImportError:
    _HAS_SUMMARIZER = False

try:
    from src.rag import (
        build_burst_rag, retrieve_for_spike,
        format_rag_context, burst_index_exists,
    )
    from src.embeddings import is_available as embeddings_available
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False

try:
    from src.qa import answer_question, build_event_context
    _HAS_QA = True
except ImportError:
    _HAS_QA = False

try:
    from src.compare import (
        get_historical_spikes, compute_spike_stats,
        compute_historical_stats_bulk, compare_spikes,
        generate_comparison_narrative, format_comparison_table,
    )
    _HAS_COMPARE = True
except ImportError:
    _HAS_COMPARE = False

try:
    from src.export_pdf import generate_briefing_pdf, is_available as pdf_available
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    from src.keywords import top_keywords, build_text_field, COUNTRY_EXTRA_STOPS
    from src.utils import load_vectorizer
    _HAS_KEYWORDS = True
except ImportError:
    _HAS_KEYWORDS = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════════════════════════════

try:
    df = load_events()
    default_burst_df = load_bursts()
except FileNotFoundError:
    data_not_found()

show_data_window()
countries = sidebar_country_filter(default_burst_df, key="burst_countries")

st.sidebar.divider()
st.sidebar.subheader("Sensitivity")
z_threshold = st.sidebar.slider(
    "Detection threshold", 1.0, 5.0, BURST_Z_THRESHOLD, 0.25, key="burst_z",
    help="Lower = more sensitive. Higher = only the strongest spikes.",
)
rolling_window = st.sidebar.slider(
    "Baseline window (days)", 3, 14, BURST_ROLLING_WINDOW, key="burst_win",
    help="Days averaged to establish the 'typical' activity level.",
)
recompute = st.sidebar.button("Recalculate", key="burst_recompute")

# ── Ollama status indicator in sidebar ───────────────────────────────────────
if _HAS_SUMMARIZER:
    st.sidebar.divider()
    ollama_ok = is_ollama_available()
    if ollama_ok:
        st.sidebar.success("Ollama available — LLM refinement active")
    else:
        st.sidebar.info("Ollama not running — template summaries active")

st.header("Activity Spikes")
st.caption("Detecting when event volumes rose significantly above baseline — and understanding why.")

if recompute:
    with st.spinner("Recalculating…"):
        burst_df = detect_bursts(df, rolling_window=rolling_window, z_threshold=z_threshold)
else:
    burst_df = default_burst_df.copy()

burst_df = burst_df[burst_df["country"].isin(countries)]

if burst_df.empty:
    empty_state("No data for selected countries.")
    st.stop()

# ── Derived values ─────────────────────────────────────────────────────────────
total_burst = int(burst_df["is_burst"].sum())
_z_clean = burst_df["z_score"].replace([np.inf, -np.inf], np.nan)
max_z = _z_clean.max()
max_z_display = f"{max_z:.1f}σ" if pd.notna(max_z) else "—"
avg_burst_count = (
    burst_df[burst_df["is_burst"]]["event_count"].mean()
    if total_burst > 0 else 0
)
burst_summary = get_burst_summary(burst_df)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_summary, tab_countries, tab_details, tab_intel = st.tabs([
    "Summary", "Country Analysis", "Spike Details", "Intelligence Hub"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Summary (unchanged)
# ════════════════════════════════════════════════════════════════════════════
with tab_summary:
    metric_row([
        ("Spike Days", str(total_burst)),
        ("Days Analysed", str(burst_df["day"].nunique())),
        ("Strongest Spike", max_z_display),
        ("Avg Events on Spike Day", f"{avg_burst_count:,.0f}" if total_burst > 0 else "—"),
    ])
    st.markdown("")

    if len(countries) > 1 and total_burst > 0:
        st.subheader("Cross-Country Spike Patterns")
        burst_days_per_country = {
            c: set(
                burst_df[(burst_df["country"] == c) & burst_df["is_burst"]]["day"]
                .dt.strftime("%Y-%m-%d")
            )
            for c in countries
        }
        all_burst_days = set()
        for days in burst_days_per_country.values():
            all_burst_days |= days

        overlaps = [
            {"Date": day, "Countries": ", ".join(
                c for c, days in burst_days_per_country.items() if day in days
            )}
            for day in sorted(all_burst_days)
            if sum(1 for days in burst_days_per_country.values() if day in days) > 1
        ]
        if overlaps:
            st.warning(
                f"{len(overlaps)} day{'s' if len(overlaps) > 1 else ''} with simultaneous "
                f"spikes across countries — possible shared triggers."
            )
            st.dataframe(pd.DataFrame(overlaps), use_container_width=True, hide_index=True)
        else:
            st.caption("No simultaneous spikes detected across countries in this window.")

    st.markdown("")
    st.subheader("Event Composition During Spikes")
    burst_days_set = burst_df[burst_df["is_burst"]][["day", "country"]]
    if not burst_days_set.empty:
        burst_events = df.merge(burst_days_set, on=["day", "country"], how="inner")
        if not burst_events.empty and "QuadLabel" in burst_events.columns:
            quad_burst = (
                burst_events.groupby(["country", "QuadLabel"])
                .size().reset_index(name="Events")
                .rename(columns={"country": "Country", "QuadLabel": "Event Class"})
            )
            fig_q = px.bar(
                quad_burst, x="Country", y="Events", color="Event Class",
                barmode="stack",
                color_discrete_map={
                    "Verbal Cooperation": "#00CC96", "Material Cooperation": "#19D3F3",
                    "Verbal Conflict": "#FFA15A", "Material Conflict": "#EF553B",
                },
            )
            fig_q.update_layout(height=340, margin=dict(t=20),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.caption("No spike event data available.")
    else:
        st.caption("No spike days detected with current settings.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Country Analysis (unchanged)
# ════════════════════════════════════════════════════════════════════════════
with tab_countries:
    for country in sorted(countries):
        cdf = burst_df[burst_df["country"] == country].copy().sort_values("day")
        if cdf.empty:
            continue
        fig = go.Figure()
        upper = cdf["rolling_mean"] + 2 * cdf["rolling_std"]
        lower = (cdf["rolling_mean"] - 2 * cdf["rolling_std"]).clip(lower=0)
        fig.add_trace(go.Scatter(x=cdf["day"], y=upper, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cdf["day"], y=lower, mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(171,99,250,0.07)",
                                 showlegend=False, hoverinfo="skip", name="Expected range"))
        fig.add_trace(go.Scatter(x=cdf["day"], y=cdf["event_count"],
                                 mode="lines+markers", name="Events",
                                 line=dict(color=COLOR_MAP_COUNTRY.get(country, "#636EFA"), width=2),
                                 marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=cdf["day"], y=cdf["rolling_mean"],
                                 mode="lines", name="Baseline",
                                 line=dict(color="#AB63FA", dash="dash", width=1.5)))
        bursts = cdf[cdf["is_burst"]]
        if not bursts.empty:
            fig.add_trace(go.Scatter(
                x=bursts["day"], y=bursts["event_count"],
                mode="markers", name="Spike",
                marker=dict(color="red", size=8, symbol="circle",
                            line=dict(width=1.5, color="darkred")),
                customdata=np.column_stack([bursts["day"].dt.strftime("%Y-%m-%d"),
                                            bursts["z_score"].round(1)]),
                hovertemplate="%{customdata[0]}  ·  %{y:,} events  ·  %{customdata[1]}σ<extra>Spike</extra>",
            ))
        fig.update_layout(title=country, height=340, xaxis_title="", yaxis_title="Events",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          margin=dict(t=45, b=20), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spike Details (unchanged)
# ════════════════════════════════════════════════════════════════════════════
with tab_details:
    if burst_summary.empty:
        st.caption("No spike days detected. Try lowering the detection threshold.")
    else:
        st.subheader("Spike Days")
        st.caption("Select a spike day to investigate in the Event Chain Explorer.")
        for _, row in burst_summary.iterrows():
            day_str  = row["day"].strftime("%Y-%m-%d")
            country  = row["country"]
            z        = row["z_score"]
            events   = int(row["event_count"])
            severity = "High" if z >= 3 else "Moderate"
            severity_color = "#EF553B" if z >= 3 else "#FFA15A"
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"**{day_str}** · {country} · {events:,} events · "
                    f"{z:.1f}σ · "
                    f"<span style='color:{severity_color};font-weight:600;'>{severity}</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Investigate →", key=f"spike_btn_{day_str}_{country}",
                             help=f"Open Event Chain for {country} on {day_str}"):
                    st.session_state["selected_date"]    = day_str
                    st.session_state["selected_country"] = country
                    st.switch_page("pages/7_Event_Chain.py")
        st.markdown("")
        st.divider()
        st.subheader("Spike Day Details")
        display = burst_summary.copy()
        display["day"] = display["day"].dt.strftime("%Y-%m-%d")
        display["Severity"] = display["z_score"].apply(lambda z: "High" if z >= 3 else "Moderate")
        display = display.rename(columns={"day": "Date", "country": "Country",
                                          "event_count": "Events",
                                          "rolling_mean": "Baseline (avg)",
                                          "z_score": "Strength (σ)"})
        display["Baseline (avg)"] = pd.to_numeric(display["Baseline (avg)"], errors="coerce").fillna(0).round(0).astype(int)
        display["Strength (σ)"]   = display["Strength (σ)"].round(1)
        st.dataframe(display[["Date","Country","Events","Baseline (avg)","Strength (σ)","Severity"]],
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Intelligence Hub  (NEW)
# ════════════════════════════════════════════════════════════════════════════
with tab_intel:
    st.subheader("Intelligence Hub")
    st.caption(
        "Select a detected spike to generate an analyst briefing, retrieve "
        "news evidence, ask questions, and compare with past events."
    )

    if burst_summary.empty:
        st.info("No spikes detected with current settings. Lower the detection threshold.")
        st.stop()

    # ── Spike selector ────────────────────────────────────────────────────────
    spike_options = {
        f"{row['day'].strftime('%Y-%m-%d')}  ·  {row['country']}  ·  "
        f"{row['z_score']:.1f}σ  ·  {int(row['event_count']):,} events": (
            row["day"].strftime("%Y-%m-%d"), row["country"]
        )
        for _, row in burst_summary.iterrows()
    }
    selected_label = st.selectbox(
        "Select spike to analyse",
        list(spike_options.keys()),
        key="intel_spike_selector",
    )
    sel_date, sel_country = spike_options[selected_label]
    sel_row = burst_summary[
        (burst_summary["day"] == pd.Timestamp(sel_date)) &
        (burst_summary["country"] == sel_country)
    ].iloc[0]

    st.markdown(
        f"**Analysing:** {sel_country}  ·  {sel_date}  ·  "
        f"{int(sel_row['event_count']):,} events  ·  {sel_row['z_score']:.1f}σ"
    )
    st.divider()

    # ── Helper: get keywords for spike ───────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _spike_keywords(date_str, country, _df=None):
        if not _HAS_KEYWORDS:
            return []
        try:
            vec = load_vectorizer()
            day_ts = pd.Timestamp(date_str)
            day_mask = (df["day"] == day_ts) & (df["country"] == country)
            day_df   = df[day_mask]
            if day_df.empty:
                return []
            texts = build_text_field(day_df)
            extra = COUNTRY_EXTRA_STOPS.get(country, set())
            kws = top_keywords(texts, vec, top_n=10, extra_stops=extra)
            return [k["keyword"] for k in kws]
        except Exception:
            return []

    kw_list = _spike_keywords(sel_date, sel_country)

    # ── Build spike_info dict ─────────────────────────────────────────────────
    spike_info = {}
    if _HAS_SUMMARIZER:
        spike_info = spike_info_from_row(sel_row, df=df, burst_df=burst_df)

    # ════════════════════════════════════════════════════════════════════════
    # Section A: Summarise Spike
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### Analyst Summary")
    if not _HAS_SUMMARIZER:
        st.warning("Install `requirements.txt` to enable summarization.")
    else:
        use_ollama = st.checkbox("Use Ollama refinement (if available)", value=True, key="intel_ollama")
        gen_summary = st.button("Summarise This Spike", key="intel_summarise", type="primary")

        if gen_summary or st.session_state.get(f"summary_{sel_date}_{sel_country}"):
            with st.spinner("Generating analyst briefing…"):
                # Get RAG chunks if available
                rag_chunks_for_summary = []
                if _HAS_RAG and burst_index_exists(sel_date, sel_country):
                    rag_chunks_for_summary = retrieve_for_spike(
                        f"geopolitical events {sel_country} {sel_date}", sel_date, sel_country, top_k=3
                    )

                event_type_dist = get_event_type_dist(df, sel_date, sel_country)
                summary_text = summarize_spike(
                    spike_info=spike_info,
                    keywords=kw_list,
                    rag_chunks=rag_chunks_for_summary,
                    event_type_dist=event_type_dist,
                    use_ollama=use_ollama,
                )
                st.session_state[f"summary_{sel_date}_{sel_country}"] = summary_text

        cached_summary = st.session_state.get(f"summary_{sel_date}_{sel_country}", "")
        if cached_summary:
            st.info(cached_summary)
        else:
            st.caption("Click 'Summarise This Spike' to generate an analyst briefing.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # Section B: Evidence Retrieval (RAG)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### Evidence from News Sources")

    if not _HAS_RAG:
        st.warning(
            "RAG evidence not available. Install: "
            "`pip install sentence-transformers faiss-cpu trafilatura beautifulsoup4`"
        )
    else:
        emb_ok, emb_reason = embeddings_available()
        if not emb_ok:
            st.warning(f"Embedding engine unavailable: {emb_reason}")
        else:
            index_exists = burst_index_exists(sel_date, sel_country)
            col_ev1, col_ev2 = st.columns([3, 1])
            with col_ev1:
                if index_exists:
                    st.success(f"Evidence index cached for {sel_country} {sel_date}.")
                else:
                    st.caption("No evidence cached yet. Click 'Fetch Evidence' to retrieve articles.")
            with col_ev2:
                fetch_btn = st.button(
                    "Fetch Evidence" if not index_exists else "Refresh Evidence",
                    key="intel_fetch_rag",
                )

            if fetch_btn:
                status_box = st.empty()
                def _rag_progress(msg):
                    status_box.info(msg)
                with st.spinner("Fetching and indexing articles…"):
                    idx, meta, articles = build_burst_rag(
                        df, sel_date, sel_country,
                        max_articles=10, force_rebuild=fetch_btn,
                        progress_callback=_rag_progress,
                    )
                    st.session_state[f"rag_meta_{sel_date}_{sel_country}"] = meta
                    st.session_state[f"rag_articles_{sel_date}_{sel_country}"] = articles
                status_box.empty()
                if meta:
                    st.success(f"Indexed {len(meta)} text chunks from {len(articles)} articles.")
                else:
                    st.warning("No article text could be retrieved. URLs may be paywalled or unavailable.")

            # Show evidence if index exists
            if burst_index_exists(sel_date, sel_country):
                with st.expander("View Retrieved Evidence", expanded=False):
                    evidence_query = st.text_input(
                        "Search evidence",
                        value=f"{sel_country} geopolitical events {sel_date}",
                        key="intel_ev_query",
                    )
                    evidence_chunks = retrieve_for_spike(
                        evidence_query, sel_date, sel_country, top_k=5
                    )
                    if evidence_chunks:
                        for i, chunk in enumerate(evidence_chunks, 1):
                            domain = chunk.get("domain", chunk.get("source_url", "?")[:40])
                            score  = chunk.get("score", 0)
                            text   = chunk.get("text", "")
                            url    = chunk.get("source_url", "")
                            st.markdown(
                                f"**[{i}] {domain}** — relevance: `{score:.3f}`  \n"
                                f"[{url[:70]}]({url})"
                            )
                            st.caption(text[:400] + ("…" if len(text) > 400 else ""))
                            st.markdown("---")
                    else:
                        st.caption("No chunks retrieved. Try a different search query.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # Section C: Q&A Interface
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### Ask a Question")
    st.caption(
        "Ask anything about this spike. The system retrieves relevant event "
        "data and news evidence to answer."
    )

    if not _HAS_QA:
        st.warning("Q&A module not available. Check src/qa.py installation.")
    else:
        # Initialise conversation history
        chat_key = f"chat_{sel_date}_{sel_country}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        # Display conversation history
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_question = st.chat_input(
            f"Ask about {sel_country} on {sel_date}…",
            key="intel_chat_input",
        )

        if user_question:
            # Display user message
            st.session_state[chat_key].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer…"):
                    # Get RAG chunks for this question
                    qa_rag_chunks = []
                    if _HAS_RAG and burst_index_exists(sel_date, sel_country):
                        qa_rag_chunks = retrieve_for_spike(
                            user_question, sel_date, sel_country, top_k=4
                        )

                    answer = answer_question(
                        question=user_question,
                        date_str=sel_date,
                        country=sel_country,
                        df=df,
                        spike_info=spike_info,
                        rag_chunks=qa_rag_chunks,
                        history=st.session_state[chat_key][:-1],
                        use_ollama=True,
                    )
                st.markdown(answer)
                st.session_state[chat_key].append({"role": "assistant", "content": answer})

        if st.session_state[chat_key]:
            if st.button("Clear conversation", key="intel_clear_chat"):
                st.session_state[chat_key] = []
                st.rerun()

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # Section D: Compare with Past Spikes
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### Compare with Past Spikes")
    st.caption(
        f"Compares this {sel_country} spike against the previous "
        "5 detected bursts for the same country."
    )

    if not _HAS_COMPARE:
        st.warning("Comparison module not available. Check src/compare.py.")
    else:
        n_hist = st.slider("Historical spikes to compare", 2, 8, 5, key="intel_n_hist")
        run_compare = st.button("Compare Spikes", key="intel_compare")

        compare_key = f"compare_{sel_date}_{sel_country}_{n_hist}"

        if run_compare or st.session_state.get(compare_key):
            if run_compare:
                with st.spinner("Computing spike statistics…"):
                    hist_spikes = get_historical_spikes(
                        burst_df, sel_country, sel_date, n=n_hist
                    )
                    current_stats = compute_spike_stats(df, burst_df, sel_date, sel_country)
                    hist_stats    = compute_historical_stats_bulk(df, burst_df, hist_spikes, sel_country)
                    comparison    = compare_spikes(current_stats, hist_stats)
                    narrative     = generate_comparison_narrative(comparison, sel_country)
                    st.session_state[compare_key] = {
                        "current": current_stats,
                        "hist_stats": hist_stats,
                        "comparison": comparison,
                        "narrative": narrative,
                    }

            cached_compare = st.session_state.get(compare_key, {})
            if cached_compare:
                comp = cached_compare["comparison"]
                narrative = cached_compare["narrative"]

                # Narrative
                st.info(narrative)

                # Metrics highlight row
                if comp.get("has_history"):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric(
                        "Size vs. avg",
                        f"{comp['count_ratio']:.1f}×",
                        delta=f"{(comp['count_ratio']-1)*100:+.0f}%",
                    )
                    col_b.metric(
                        "Z-score vs. avg",
                        f"{comp['z_ratio']:.1f}×",
                    )
                    col_c.metric(
                        "Tone vs. avg",
                        f"{comp['tone_diff']:+.2f}",
                        delta="more negative" if comp["tone_diff"] < 0 else "more positive",
                        delta_color="inverse" if comp["tone_diff"] < 0 else "normal",
                    )
                    col_d.metric(
                        "Pattern",
                        comp["escalation_direction"],
                    )

                # Comparison table
                if cached_compare.get("hist_stats"):
                    table = format_comparison_table(
                        cached_compare["current"],
                        cached_compare["hist_stats"],
                        comp,
                    )
                    st.dataframe(table, use_container_width=True, hide_index=True)
                else:
                    st.caption("No historical spikes found for comparison.")
        else:
            st.caption("Click 'Compare Spikes' to run the comparative analysis.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # Section E: Export PDF Briefing
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### Export Analyst Briefing")
    st.caption("Download a one-page PDF intelligence briefing for this spike.")

    if not _HAS_PDF or not pdf_available():
        st.warning("PDF export not available. Install: `pip install fpdf2`")
    else:
        pdf_btn = st.button("Generate Briefing PDF", key="intel_pdf", type="secondary")

        if pdf_btn:
            # Gather all available data for the PDF
            cached_summary   = st.session_state.get(f"summary_{sel_date}_{sel_country}", "")
            cached_compare_d = st.session_state.get(
                f"compare_{sel_date}_{sel_country}_{st.session_state.get('intel_n_hist', 5)}", {}
            )
            comparison_note = (
                cached_compare_d.get("narrative", "")
                if cached_compare_d else ""
            )

            # Get evidence URLs
            evidence_urls = []
            if _HAS_RAG and burst_index_exists(sel_date, sel_country):
                ev_chunks = retrieve_for_spike(
                    f"{sel_country} events {sel_date}", sel_date, sel_country, top_k=6
                )
                seen_domains = set()
                for c in ev_chunks:
                    url = c.get("source_url", "")
                    domain = c.get("domain", "")
                    if url and domain not in seen_domains:
                        evidence_urls.append(url)
                        seen_domains.add(domain)

            # Compute avg tone
            day_ts = pd.Timestamp(sel_date)
            day_mask = (df["day"] == day_ts) & (df["country"] == sel_country)
            avg_tone_val = float(df[day_mask]["AvgTone"].mean()) if not df[day_mask].empty else 0.0
            conflict_pct = 0.0
            if not df[day_mask].empty and "QuadClass" in df.columns:
                conflict_pct = 100 * (df[day_mask]["QuadClass"] >= 3).mean()

            spike_data_for_pdf = {
                "country"     : sel_country,
                "date_str"    : sel_date,
                "event_count" : int(sel_row["event_count"]),
                "z_score"     : float(sel_row["z_score"]),
                "baseline"    : float(sel_row.get("rolling_mean", 0)),
                "avg_tone"    : avg_tone_val,
                "conflict_pct": conflict_pct,
            }

            with st.spinner("Generating PDF…"):
                try:
                    pdf_bytes = generate_briefing_pdf(
                        spike_data=spike_data_for_pdf,
                        summary=cached_summary or (
                            f"Statistical anomaly detected in {sel_country} on {sel_date}. "
                            f"Event count: {int(sel_row['event_count']):,}. "
                            f"Z-score: {float(sel_row['z_score']):.2f}σ. "
                            "Generate the analyst summary above for a detailed briefing."
                        ),
                        keywords=kw_list,
                        evidence_urls=evidence_urls,
                        comparison_note=comparison_note,
                    )
                    filename = f"GDELT_Briefing_{sel_country}_{sel_date}.pdf"
                    st.download_button(
                        label="Download PDF Briefing",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key="intel_pdf_download",
                    )
                    st.success(f"PDF ready: {filename}")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
