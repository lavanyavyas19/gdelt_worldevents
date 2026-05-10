
from __future__ import annotations

import os
import json
import textwrap
from typing import Any, Dict, List, Optional


try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODEL)

# Max tokens for each call type
_TOKENS_SUMMARY  = 350
_TOKENS_CHAIN    = 400
_TOKENS_QA       = 500
_TOKENS_COMPARE  = 500
_TOKENS_BRIEFING = 700




def _client() -> "anthropic.Anthropic":
    """Return an Anthropic client, raising a clear error if not configured."""
    if not _HAS_ANTHROPIC:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it in your terminal: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def _call(system: str, user: str, max_tokens: int) -> str:
    """Make a single Claude API call and return the text response."""
    client = _client()
    msg = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


def _fallback_msg(error: Exception) -> str:
    """Return a user-visible fallback when the API call fails."""
    err_str = str(error)
    if "ANTHROPIC_API_KEY" in err_str:
        return (
            "⚠️ **API key not configured.** "
            "Set the `ANTHROPIC_API_KEY` environment variable to enable AI summaries."
        )
    if "anthropic package" in err_str:
        return (
            "⚠️ **anthropic package not installed.** "
            "Run `pip install anthropic` and restart the app."
        )
    return f"⚠️ **AI summary unavailable:** {err_str[:200]}"




_SYSTEM_ANALYST = textwrap.dedent("""
    You are a senior geopolitical analyst at a tier-1 intelligence consultancy.
    You specialise in event-data analytics, bilateral relations, and pattern
    recognition across open-source data feeds (GDELT, ACLED, etc.).

    Communication style:
    - Precise, confident, analytical. No filler phrases.
    - Use specific numbers from the context; never fabricate statistics.
    - Write in plain English — no jargon without explanation.
    - Avoid starting sentences with "This", "The data shows", or "Notably".
    - Do not use bullet points unless explicitly asked.
    - Maximum 5 sentences unless instructed otherwise.
""").strip()

_SYSTEM_QA = textwrap.dedent("""
    You are an expert geopolitical analyst with access to structured GDELT
    event data. Answer questions accurately based ONLY on the context provided.
    If the context is insufficient, say so clearly — do not speculate.

    Rules:
    - Ground every claim in the provided data context.
    - If you cite a statistic, it must appear in the context.
    - Use professional, direct language.
    - Keep answers under 150 words unless the question requires more detail.
""").strip()



def summarize_spike(
    country: str,
    date_str: str,
    event_count: int,
    baseline: float,
    z_score: float,
    tone_delta: float,
    top_keywords: List[str],
    chain_events: Optional[List[Dict]] = None,
    rag_context: str = "",
) -> str:
    """
    Generate a 3–5 sentence analyst briefing for a detected activity spike.

    Parameters
    ----------
    country       : Country name (e.g. "Iran")
    date_str      : Spike date as "YYYY-MM-DD"
    event_count   : Events on spike day
    baseline      : 7-day rolling mean events
    z_score       : Statistical anomaly score (σ above baseline)
    tone_delta    : Change in avg event tone vs. baseline
    top_keywords  : Top TF-IDF keywords from the spike period
    chain_events  : Optional list of event dicts from the chain scorer
    rag_context   : Retrieved news article excerpts (from RAG pipeline)

    Returns
    -------
    Analyst briefing as a plain-text string (Markdown-friendly).
    """

    chain_summary = ""
    if chain_events:
        recent = chain_events[:5]
        chain_summary = "; ".join(
            f"{e.get('event_date','?')} — {e.get('EventRootCode','?')}"
            f" ({e.get('QuadLabel', e.get('EventClass','?'))})"
            for e in recent
        )

 
    rag_block = ""
    if rag_context.strip():
        rag_block = f"\n\nNews evidence (retrieved):\n{rag_context[:1200]}"

    user_prompt = textwrap.dedent(f"""
        A statistical anomaly was detected in geopolitical event data.

        Country:        {country}
        Date:           {date_str}
        Event count:    {event_count:,}  (baseline: {baseline:.0f})
        Z-score:        {z_score:.2f}σ
        Tone shift:     {tone_delta:+.2f} points vs. baseline
        Top keywords:   {", ".join(top_keywords[:10]) if top_keywords else "N/A"}
        Preceding events: {chain_summary or "Not available"}{rag_block}

        Write a 3–5 sentence intelligence briefing:
        Sentence 1: What happened (what the spike represents).
        Sentence 2: The statistical significance (interpret z-score and tone shift
                    in plain language — do NOT just repeat the numbers).
        Sentence 3: What the keywords reveal about the dominant themes.
        Sentence 4 (optional): What pattern the preceding events suggest.
        Sentence 5 (optional): What to watch in the coming days.

        Do NOT use bullet points. Write as flowing prose.
    """).strip()

    try:
        return _call(_SYSTEM_ANALYST, user_prompt, _TOKENS_SUMMARY)
    except Exception as e:
        return _fallback_msg(e)


def explain_chain(
    country: str,
    anchor_event: Dict,
    before_events: List[Dict],
    after_events: List[Dict],
    pattern: str,
    rag_context: str = "",
) -> str:
    """
    Generate a narrative explanation of an event chain.

    Parameters
    ----------
    country       : Country context
    anchor_event  : The focal event (dict with event_date, EventRootCode, etc.)
    before_events : Events preceding the anchor (chronologically)
    after_events  : Events following the anchor
    pattern       : Detected pattern label (e.g. "Escalation")
    rag_context   : Retrieved news excerpts

    Returns
    -------
    Chain narrative as a plain-text string.
    """
    def _fmt_events(evts: List[Dict], label: str) -> str:
        if not evts:
            return f"{label}: None retrieved"
        lines = []
        for e in evts[:5]:
            dt   = e.get("event_date", "?")
            code = e.get("EventRootCode", "?")
            quad = e.get("QuadLabel", e.get("EventClass", "?"))
            tone = e.get("AvgTone", None)
            tone_str = f" | tone {tone:.1f}" if tone is not None else ""
            lines.append(f"  • {dt}: {code} [{quad}]{tone_str}")
        return f"{label}:\n" + "\n".join(lines)

    anchor_desc = (
        f"{anchor_event.get('event_date','?')} | "
        f"{anchor_event.get('EventRootCode','?')} | "
        f"{anchor_event.get('QuadLabel', anchor_event.get('EventClass','?'))} | "
        f"tone {anchor_event.get('AvgTone', 'N/A')}"
    )

    rag_block = ""
    if rag_context.strip():
        rag_block = f"\n\nNews context:\n{rag_context[:800]}"

    user_prompt = textwrap.dedent(f"""
        Analyse this geopolitical event chain from {country}.

        Anchor event: {anchor_desc}
        Pattern classification: {pattern}

        {_fmt_events(before_events, "Preceding events (oldest → newest)")}
        {_fmt_events(after_events, "Subsequent events")}
        {rag_block}

        Write a 3–4 sentence narrative that:
        1. Describes what the anchor event represents in plain language.
        2. Explains how the preceding events led up to it.
        3. Interprets the detected pattern ({pattern}) — what it implies about
           the trajectory of this situation.
        4. Identifies the most analytically significant signal in this chain.

        Write as flowing prose. No bullet points.
    """).strip()

    try:
        return _call(_SYSTEM_ANALYST, user_prompt, _TOKENS_CHAIN)
    except Exception as e:
        return _fallback_msg(e)


def answer_analyst_query(
    question: str,
    event_context: str,
    rag_context: str = "",
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """
    Answer a natural language question about the geopolitical data.

    Parameters
    ----------
    question            : User's question
    event_context       : Structured event data as text (burst stats, keywords, etc.)
    rag_context         : Retrieved news article excerpts
    conversation_history: Previous messages [{"role": "user"|"assistant", "content": "..."}]

    Returns
    -------
    Analytical answer as a plain-text string.
    """
    context_block = f"Structured event data:\n{event_context}"
    if rag_context.strip():
        context_block += f"\n\nRetrieved news evidence:\n{rag_context[:1000]}"

    user_content = f"{context_block}\n\nQuestion: {question}"

    try:
        if not _HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed.")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")

        client = anthropic.Anthropic(api_key=api_key)

        # Build messages with history
        messages: List[Dict] = []
        if conversation_history:
            messages.extend(conversation_history[-6:])  # last 6 turns max
        messages.append({"role": "user", "content": user_content})

        msg = client.messages.create(
            model=_MODEL,
            max_tokens=_TOKENS_QA,
            system=_SYSTEM_QA,
            messages=messages,
        )
        return msg.content[0].text.strip()

    except Exception as e:
        return _fallback_msg(e)


def compare_countries(
    country_stats: Dict[str, Dict],
    burst_counts: Dict[str, int],
    top_keywords: Dict[str, List[str]],
    date_range: str,
) -> str:
    """
    Generate a comparative narrative across country datasets.

    Parameters
    ----------
    country_stats  : {country: {"event_count": int, "avg_tone": float,
                                "conflict_pct": float}}
    burst_counts   : {country: n_burst_days}
    top_keywords   : {country: [keyword, ...]}
    date_range     : Human-readable date range string

    Returns
    -------
    Comparative analysis paragraph.
    """
    stats_lines = []
    for country, stats in country_stats.items():
        keywords = ", ".join(top_keywords.get(country, [])[:5])
        stats_lines.append(
            f"{country}: {stats.get('event_count',0):,} events | "
            f"avg tone {stats.get('avg_tone',0):.2f} | "
            f"{stats.get('conflict_pct',0):.1f}% conflict events | "
            f"{burst_counts.get(country, 0)} burst days | "
            f"top keywords: {keywords or 'N/A'}"
        )

    user_prompt = textwrap.dedent(f"""
        Compare geopolitical event patterns across these countries ({date_range}):

        {chr(10).join(stats_lines)}

        Write a 4-sentence comparative analysis:
        1. Identify which country shows the highest activity and what type.
        2. Compare tone profiles — what do they reveal about relationship dynamics?
        3. Compare burst frequencies — which country is most volatile?
        4. Identify the most analytically interesting contrast between the countries.

        Write as flowing prose. No bullet points. Be specific with numbers.
    """).strip()

    try:
        return _call(_SYSTEM_ANALYST, user_prompt, _TOKENS_COMPARE)
    except Exception as e:
        return _fallback_msg(e)


def generate_briefing(
    country: str,
    date_str: str,
    spike_summary: str,
    chain_narrative: str,
    top_keywords: List[str],
    z_score: float,
    event_count: int,
    tone_avg: float,
) -> str:
    """
    Generate a full structured intelligence briefing for a spike event.
    Suitable for export or display as a standalone report.

    Returns
    -------
    Full briefing as Markdown-formatted text.
    """
    user_prompt = textwrap.dedent(f"""
        Generate a structured intelligence briefing for the following event.

        Country: {country}
        Date: {date_str}
        Event count: {event_count:,}  |  Z-score: {z_score:.2f}σ  |  Avg tone: {tone_avg:.2f}
        Top keywords: {", ".join(top_keywords[:10])}

        Spike summary (pre-generated): {spike_summary}
        Chain narrative (pre-generated): {chain_narrative}

        Write the briefing in this exact structure (use Markdown headers):

        ## EXECUTIVE SUMMARY
        (2 sentences: what happened and why it matters)

        ## STATISTICAL SIGNIFICANCE
        (1–2 sentences: interpret the z-score and tone in plain language)

        ## KEY THEMES
        (2–3 sentences: what the keywords reveal about the dominant discourse)

        ## CHAIN OF EVENTS
        (2–3 sentences: the sequence leading to this spike)

        ## ANALYST ASSESSMENT
        (2 sentences: implications and what to watch next)

        Keep each section tightly within the bounds above. No padding.
    """).strip()

    try:
        return _call(_SYSTEM_ANALYST, user_prompt, _TOKENS_BRIEFING)
    except Exception as e:
        return _fallback_msg(e)




def is_available() -> tuple[bool, str]:
    """
    Check whether LLM features are available.

    Returns
    -------
    (True, "")              if fully configured
    (False, reason_string)  if unavailable
    """
    if not _HAS_ANTHROPIC:
        return False, "anthropic package not installed (run: pip install anthropic)"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return False, "ANTHROPIC_API_KEY environment variable not set"
    return True, ""
