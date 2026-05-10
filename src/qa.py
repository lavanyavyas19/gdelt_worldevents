
from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Optional, Tuple

import pandas as pd



from .summarizer import is_ollama_available, refine_with_ollama




_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("spike_cause", re.compile(
        r"\b(why|cause|reason|trigger|drove|driving|what happened|explain)\b",
        re.I,
    )),
    ("evidence", re.compile(
        r"\b(evidence|source|article|news|report|support|proof|link|url)\b",
        re.I,
    )),
    ("comparison", re.compile(
        r"\b(compar|previous|past|before|last time|worse|stronger|bigger|differ)\b",
        re.I,
    )),
    ("keywords", re.compile(
        r"\b(keyword|topic|theme|term|tfidf|subject|about|discuss)\b",
        re.I,
    )),
    ("event_types", re.compile(
        r"\b(event type|type of event|quad|cameo|conflict|cooperation|what kind)\b",
        re.I,
    )),
    ("tone", re.compile(
        r"\b(tone|sentiment|positive|negative|mood|attitude|hostile)\b",
        re.I,
    )),
    ("actors", re.compile(
        r"\b(actor|who|entity|government|person|player|involved|participant)\b",
        re.I,
    )),
    ("prediction", re.compile(
        r"\b(predict|forecast|next|future|will|expect|upcoming|continue)\b",
        re.I,
    )),
]


def parse_intent(question: str) -> str:
    """
    Classify a user question into one of the intent categories.

    Returns
    -------
    Intent string: one of the keys in _INTENT_PATTERNS, or "general".
    """
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(question):
            return intent
    return "general"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTEXT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_event_context(
    df: pd.DataFrame,
    date_str: str,
    country: str,
    burst_stats: Optional[Dict] = None,
) -> str:
    """
    Build a structured text block summarising events on a burst day.

    Parameters
    ----------
    df          : Full events DataFrame
    date_str    : "YYYY-MM-DD" burst date
    country     : Country name
    burst_stats : Optional dict from spike_info_from_row()

    Returns
    -------
    Multi-line context string for use in answer composition.
    """
    day_ts = pd.Timestamp(date_str)
    mask = (df["day"] == day_ts) & (df["country"] == country)
    day_df = df[mask].copy()

    lines = [f"[Event Context: {country} on {date_str}]"]

    if burst_stats:
        lines.append(
            f"Event count: {burst_stats.get('event_count', 0):,}  "
            f"(baseline: {burst_stats.get('baseline', 0):.0f},  "
            f"z-score: {burst_stats.get('z_score', 0):.2f}σ)"
        )
        if burst_stats.get("tone_delta") is not None:
            td = burst_stats["tone_delta"]
            lines.append(f"Tone shift vs. baseline: {td:+.2f} points")

    if day_df.empty:
        lines.append("No event records found for this date/country.")
        return "\n".join(lines)

    # Event type distribution
    if "QuadLabel" in day_df.columns:
        quad_counts = day_df["QuadLabel"].value_counts().head(4)
        lines.append("\nEvent type breakdown:")
        for label, cnt in quad_counts.items():
            pct = 100 * cnt / len(day_df)
            lines.append(f"  {label}: {cnt:,} ({pct:.0f}%)")

    # Top CAMEO event codes
    if "EventRootLabel" in day_df.columns:
        code_counts = day_df["EventRootLabel"].value_counts().head(5)
        lines.append("\nTop event categories:")
        for label, cnt in code_counts.items():
            lines.append(f"  {label}: {cnt:,}")

    # Tone statistics
    if "AvgTone" in day_df.columns:
        avg_tone = day_df["AvgTone"].mean()
        min_tone = day_df["AvgTone"].min()
        max_tone = day_df["AvgTone"].max()
        lines.append(
            f"\nTone — mean: {avg_tone:.2f},  "
            f"range: [{min_tone:.1f}, {max_tone:.1f}]"
        )

    # Top actors
    for col in ("Actor1Name", "actor1_clean"):
        if col in day_df.columns:
            top_actors = (
                day_df[col].dropna()
                .value_counts().head(5)
            )
            if not top_actors.empty:
                lines.append("\nTop Actor 1 entities:")
                for name, cnt in top_actors.items():
                    lines.append(f"  {name}: {cnt:,} events")
            break

    # Top source domains
    if "source_domain" in day_df.columns:
        top_domains = day_df["source_domain"].dropna().value_counts().head(5)
        if not top_domains.empty:
            lines.append("\nTop news sources:")
            for dom, cnt in top_domains.items():
                lines.append(f"  {dom}: {cnt:,} articles")

    return "\n".join(lines)


def _get_spike_keywords(df: pd.DataFrame, date_str: str, country: str) -> List[str]:
    """Return TF-IDF top keywords for this spike if the vectorizer is available."""
    try:
        from .keywords import get_spike_keywords
        day_ts = pd.Timestamp(date_str)
        mask = (df["day"] == day_ts) & (df["country"] == country)
        day_df = df[mask]
        if day_df.empty:
            return []
        return get_spike_keywords(df, day_df, country, top_n=8)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT-SPECIFIC ANSWER TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

def _answer_spike_cause(
    event_context: str, rag_chunks: List[Dict], keywords: List[str], spike_info: Dict
) -> str:
    country    = spike_info.get("country", "the country")
    date_str   = spike_info.get("date_str", "this period")
    z          = spike_info.get("z_score", 0)
    tone_delta = spike_info.get("tone_delta", 0)

    kw_str = ", ".join(f'"{k}"' for k in keywords[:5]) if keywords else "unavailable"
    tone_dir = "negative" if tone_delta < -0.5 else ("positive" if tone_delta > 0.5 else "neutral")

    answer = (
        f"The {z:.1f}σ burst in {country} on {date_str} was characterised by "
        f"a {tone_dir} tone shift ({tone_delta:+.2f} points) and elevated event frequency.\n\n"
        f"Key themes extracted from the event corpus: {kw_str}.\n\n"
        f"{event_context}"
    )

    if rag_chunks:
        answer += "\n\nSupporting evidence from news sources:\n"
        for chunk in rag_chunks[:2]:
            domain = chunk.get("domain", chunk.get("source_url", "unknown")[:30])
            text   = chunk.get("text", "")[:250]
            answer += f"\n[{domain}] {text}...\n"

    return answer


def _answer_evidence(rag_chunks: List[Dict]) -> str:
    if not rag_chunks:
        return (
            "No article text was retrieved for this spike. This may be because:\n"
            "• The GDELT source URLs returned no accessible content.\n"
            "• The article fetcher has not been run yet for this date.\n"
            "Click 'Show Evidence' to trigger article fetching."
        )

    lines = [f"Retrieved {len(rag_chunks)} relevant article passage(s):\n"]
    for i, chunk in enumerate(rag_chunks, 1):
        domain = chunk.get("domain", chunk.get("source_url", "unknown source")[:40])
        url    = chunk.get("source_url", "")
        score  = chunk.get("score", 0)
        text   = chunk.get("text", "")[:300]
        lines.append(
            f"[{i}] Source: {domain}  (relevance: {score:.2f})\n"
            f"    URL: {url}\n"
            f"    Excerpt: \"{text}...\"\n"
        )
    return "\n".join(lines)


def _answer_keywords(keywords: List[str], event_context: str) -> str:
    if keywords:
        kw_formatted = "\n".join(f"  • {k}" for k in keywords[:10])
        return (
            f"Top TF-IDF keywords for this spike period:\n{kw_formatted}\n\n"
            "These terms appear significantly more often during the burst than "
            "during baseline periods, making them the statistical fingerprint of this event."
        )
    return f"Keyword data unavailable. Event context:\n\n{event_context}"


def _answer_event_types(event_context: str, spike_info: Dict) -> str:
    return (
        f"Event type breakdown for {spike_info.get('country', 'this country')} "
        f"on {spike_info.get('date_str', 'this date')}:\n\n{event_context}"
    )


def _answer_tone(spike_info: Dict, event_context: str) -> str:
    tone_delta = spike_info.get("tone_delta", 0)
    z          = spike_info.get("z_score", 0)
    country    = spike_info.get("country", "the country")

    if tone_delta < -1.5:
        interp = (
            f"Tone was markedly negative (Δ{tone_delta:.2f} from baseline). "
            "This is consistent with conflict escalation, diplomatic deterioration, "
            "or a significant adverse event."
        )
    elif tone_delta < -0.3:
        interp = (
            f"Tone shifted mildly negative (Δ{tone_delta:.2f}), suggesting "
            "some adversarial dynamics alongside the heightened activity."
        )
    elif tone_delta > 1.0:
        interp = (
            f"Tone was positive (Δ{tone_delta:+.2f}), indicating the surge was "
            "driven by cooperative or constructive events."
        )
    else:
        interp = (
            f"Tone was broadly neutral (Δ{tone_delta:+.2f}), suggesting the "
            "spike reflects volume increase rather than a directional shift."
        )

    return f"Tone analysis for {country} (z-score: {z:.1f}σ):\n\n{interp}\n\n{event_context}"


def _answer_actors(event_context: str, spike_info: Dict) -> str:
    return (
        f"Actor analysis for {spike_info.get('country', 'this country')} "
        f"on {spike_info.get('date_str', 'this date')}:\n\n{event_context}"
    )


def _answer_prediction(spike_info: Dict) -> str:
    tone_delta = spike_info.get("tone_delta", 0)
    z          = spike_info.get("z_score", 0)
    country    = spike_info.get("country", "the country")

    if tone_delta < -1.0 and z >= 3:
        outlook = (
            "Given the combination of high z-score and negative tone shift, "
            "there is elevated risk of continued or escalated activity in the "
            "near term. Monitor for follow-on conflict-coded events."
        )
    elif tone_delta > 0.5:
        outlook = (
            "The positive tone profile suggests this may be a constructive "
            "peak rather than a conflict escalation. Activity may normalise "
            "within 7–10 days."
        )
    else:
        outlook = (
            "The statistical pattern is ambiguous. Recommendation: apply "
            "the burst detector with a 7-day forward window to assess whether "
            "activity remains elevated."
        )

    return (
        f"Forward outlook for {country} based on current spike characteristics "
        f"(z={z:.1f}σ, tone Δ{tone_delta:+.2f}):\n\n{outlook}\n\n"
        "Note: This is a heuristic projection, not a model-based forecast."
    )


def _answer_general(
    question: str, event_context: str, rag_chunks: List[Dict]
) -> str:
    answer = (
        f"Based on the structured event data for this burst period:\n\n"
        f"{event_context}\n\n"
    )
    if rag_chunks:
        answer += "Relevant news evidence:\n"
        for chunk in rag_chunks[:2]:
            text = chunk.get("text", "")[:200]
            domain = chunk.get("domain", "unknown")
            answer += f"\n[{domain}] {text}...\n"
    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN Q&A FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def answer_question(
    question: str,
    date_str: str,
    country: str,
    df: pd.DataFrame,
    spike_info: Optional[Dict] = None,
    rag_chunks: Optional[List[Dict]] = None,
    history: Optional[List[Dict]] = None,
    use_ollama: bool = True,
    ollama_model: str = "mistral",
) -> str:
    """
    Answer a natural language question about a burst event.

    Parameters
    ----------
    question    : User's free-text question
    date_str    : Burst date "YYYY-MM-DD"
    country     : Country name
    df          : Full events DataFrame
    spike_info  : Optional dict from spike_info_from_row()
    rag_chunks  : Optional List[Dict] from retrieve_for_spike()
    history     : Optional conversation history (unused in rule-based mode)
    use_ollama  : Attempt Ollama refinement if available
    ollama_model: Preferred local model

    Returns
    -------
    Answer string (Markdown-compatible).
    """
    spike_info = spike_info or {}
    rag_chunks = rag_chunks or []
    keywords   = _get_spike_keywords(df, date_str, country)

    event_context = build_event_context(df, date_str, country, spike_info)
    intent        = parse_intent(question)

    # ── Route to intent-specific handler ──────────────────────────────────────
    if intent == "spike_cause":
        raw_answer = _answer_spike_cause(event_context, rag_chunks, keywords, spike_info)
    elif intent == "evidence":
        raw_answer = _answer_evidence(rag_chunks)
    elif intent == "keywords":
        raw_answer = _answer_keywords(keywords, event_context)
    elif intent == "event_types":
        raw_answer = _answer_event_types(event_context, spike_info)
    elif intent == "tone":
        raw_answer = _answer_tone(spike_info, event_context)
    elif intent == "actors":
        raw_answer = _answer_actors(event_context, spike_info)
    elif intent == "prediction":
        raw_answer = _answer_prediction(spike_info)
    elif intent == "comparison":
        raw_answer = (
            "Use the 'Compare with Past Spikes' section above for a detailed "
            "comparison of this spike against previous events for the same country."
        )
    else:
        raw_answer = _answer_general(question, event_context, rag_chunks)

    # ── Optional Ollama refinement ────────────────────────────────────────────
    if use_ollama and is_ollama_available(ollama_model):
        prompt = textwrap.dedent(f"""
            You are an expert geopolitical analyst. A user asked:
            "{question}"

            Based on the following structured evidence, provide a clear,
            professional answer. Use only the facts provided — do not speculate.

            {raw_answer}

            Respond concisely (max 200 words). No bullet points.
        """).strip()
        refined = refine_with_ollama(prompt, preferred_model=ollama_model)
        if len(refined) > 80:
            return refined

    return raw_answer
