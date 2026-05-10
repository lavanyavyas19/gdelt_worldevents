

from __future__ import annotations

import random
import textwrap
from typing import Dict, List, Optional

import requests


_OLLAMA_URL = "http://localhost:11434"


def is_ollama_available(model: str = "mistral") -> bool:
    """
    Return True if Ollama is running and has the requested model loaded.
    Does not block — uses a 2-second timeout.
    """
    try:
        r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        return model in models or len(models) > 0   # any model is fine
    except Exception:
        return False


def _list_ollama_models() -> List[str]:
    """Return names of locally available Ollama models."""
    try:
        r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def refine_with_ollama(
    text: str,
    preferred_model: str = "mistral",
    timeout: int = 45,
) -> str:
    """
    Use a local Ollama model to lightly paraphrase/refine template text.

    Falls back to the original template text if Ollama is unavailable,
    model is not found, or request times out.

    Parameters
    ----------
    text            : Template-generated analyst text to refine
    preferred_model : Model name to try first (e.g., "mistral", "phi3")
    timeout         : HTTP request timeout in seconds

    Returns
    -------
    Refined text string (or original `text` on any failure).
    """
    available_models = _list_ollama_models()
    if not available_models:
        return text

   
    model = preferred_model
    if not any(preferred_model in m for m in available_models):
        model = available_models[0]

    prompt = textwrap.dedent(f"""
        You are a senior geopolitical analyst. Rewrite the following draft
        intelligence note to sound more professional and fluent. Keep all
        facts and numbers exactly as stated. Do not add speculation.
        Output only the rewritten text — no preamble.

        Draft:
        {text}
    """).strip()

    try:
        r = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if r.status_code == 200:
            result = r.json().get("response", "").strip()
            if len(result) > 80:    # sanity check — reject empty/tiny output
                return result
    except Exception:
        pass

    return text    # fallback to original



_S1_HIGH_Z = [
    "{country} recorded a sharp statistical anomaly on {date}, with {count:,} events "
    "— {ratio:.1f}× the recent baseline of {baseline:.0f} — triggering a {z:.1f}σ z-score.",
    "Activity in {country} surged to {count:,} events on {date}, representing a "
    "{z:.1f} standard-deviation departure from the {baseline:.0f}-event daily baseline.",
    "An acute burst of {count:,} geopolitical events was detected in {country} on "
    "{date}, a {z:.1f}σ spike above the established rolling average of {baseline:.0f}.",
]
_S1_MOD_Z = [
    "{country} saw elevated geopolitical activity on {date}, with {count:,} events "
    "representing a {z:.1f}σ increase over the recent baseline of {baseline:.0f}.",
    "Event frequency in {country} rose to {count:,} on {date} ({z:.1f}σ above the "
    "{baseline:.0f}-event average), a statistically significant but moderate anomaly.",
]


_S2_NEGATIVE = [
    "Tone metrics shifted decisively negative (Δ{tone:+.1f} points), signalling "
    "a deterioration in the diplomatic or security climate.",
    "The average event tone dropped by {tone_abs:.1f} points relative to baseline, "
    "indicating a shift toward adversarial rather than cooperative discourse.",
    "Sentiment analysis shows a {tone_abs:.1f}-point negative tone shift, consistent "
    "with escalating tensions rather than routine diplomatic exchange.",
]
_S2_POSITIVE = [
    "Tone metrics trended positively (Δ{tone:+.1f} points), suggesting cooperative "
    "or conciliatory dynamics drove the uptick.",
    "The average tone improved by {tone_abs:.1f} points, indicating the surge was "
    "driven by cooperative or diplomatic events rather than conflict.",
]
_S2_NEUTRAL = [
    "Tone remained broadly stable (Δ{tone:+.1f} points), suggesting the spike "
    "reflects volume rather than a directional shift in event character.",
]


_S3_TEMPLATES = [
    "The dominant discourse during this period centred on: {keywords}.",
    "TF-IDF keyword analysis identifies the primary themes as {keywords}.",
    "Topic modelling of spike-period events surfaces the following key terms: {keywords}.",
    "The statistical fingerprint of this period is defined by the terms {keywords}, "
    "distinguishing it from baseline weeks.",
]


_S4_CONFLICT = [
    "Conflict events constituted {conflict_pct:.0f}% of activity, with "
    "{top_type} as the dominant event class.",
    "{conflict_pct:.0f}% of events were classified as conflict-type "
    "(QuadClass 3–4), and {top_type} was the most frequently coded interaction.",
]
_S4_COOP = [
    "Activity was predominantly cooperative ({coop_pct:.0f}%), led by "
    "{top_type} events.",
    "Despite elevated volume, {coop_pct:.0f}% of events reflected cooperative "
    "dynamics, with {top_type} as the leading category.",
]
_S4_MIXED = [
    "The spike exhibited a mixed profile: {conflict_pct:.0f}% conflict events "
    "alongside {coop_pct:.0f}% cooperative, with {top_type} as the modal category.",
]


_S5_WITH_EVIDENCE = [
    "Open-source reporting from {domains} corroborates the statistical signal.",
    "Supporting evidence is drawn from coverage by {domains}.",
    "News coverage from {domains} aligns with the detected pattern.",
]
_S5_NO_EVIDENCE = [
    "No open-source article text was retrievable for this period; "
    "the assessment is based on structured event metadata alone.",
]


_S6_ESCALATION = [
    "Analysts should monitor whether this pattern extends into the following "
    "week, particularly for continued conflict-coded events.",
    "The trajectory warrants close observation over the next 7–10 days to "
    "determine whether this represents a transient shock or sustained escalation.",
]
_S6_DEESCALATION = [
    "The data suggest a possible de-escalatory dynamic; monitoring whether "
    "cooperative tones persist will be important.",
]
_S6_GENERIC = [
    "Continued monitoring of event frequency and tone in {country} is advised "
    "to assess whether this burst is a transient anomaly or part of a broader trend.",
]


def _pick(templates: List[str], seed: int = 0) -> str:
    """Pick a template quasi-randomly but deterministically per spike."""
    return templates[seed % len(templates)]



def summarize_spike(
    spike_info: Dict,
    keywords: Optional[List[str]] = None,
    rag_chunks: Optional[List[Dict]] = None,
    event_type_dist: Optional[Dict] = None,
    use_ollama: bool = True,
    ollama_model: str = "mistral",
) -> str:
    """
    Generate a 4–6 sentence analyst briefing for a detected spike.

    Parameters
    ----------
    spike_info      : Dict with keys:
                      - country (str)
                      - date_str (str, "YYYY-MM-DD")
                      - event_count (int)
                      - baseline (float)   rolling mean
                      - z_score (float)
                      - tone_delta (float) vs. baseline (negative = worse)
    keywords        : Top TF-IDF keywords for the spike period
    rag_chunks      : Retrieved article chunks from RAG (List[Dict])
    event_type_dist : {"Verbal Conflict": 320, "Material Cooperation": 150, ...}
    use_ollama      : Whether to attempt Ollama refinement
    ollama_model    : Preferred local Ollama model name

    Returns
    -------
    Multi-sentence analyst briefing string.
    """
    country    = spike_info.get("country", "the country")
    date_str   = spike_info.get("date_str", "the period")
    count      = int(spike_info.get("event_count", 0))
    baseline   = float(spike_info.get("baseline", 1))
    z          = float(spike_info.get("z_score", 0))
    tone_delta = float(spike_info.get("tone_delta", 0))

    ratio      = count / max(baseline, 1)
    tone_abs   = abs(tone_delta)
    seed       = hash(f"{country}{date_str}") % 7    # deterministic variety

    sentences = []


    s1_pool = _S1_HIGH_Z if z >= 3 else _S1_MOD_Z
    sentences.append(_pick(s1_pool, seed).format(
        country=country, date=date_str, count=count,
        baseline=baseline, z=z, ratio=ratio,
    ))


    if tone_delta < -0.5:
        s2_pool = _S2_NEGATIVE
    elif tone_delta > 0.5:
        s2_pool = _S2_POSITIVE
    else:
        s2_pool = _S2_NEUTRAL
    sentences.append(_pick(s2_pool, seed + 1).format(
        tone=tone_delta, tone_abs=tone_abs,
    ))

   
    if keywords:
        kw_str = ", ".join(f'"{k}"' for k in keywords[:6])
        sentences.append(_pick(_S3_TEMPLATES, seed + 2).format(keywords=kw_str))

    
    if event_type_dist:
        total_typed = sum(event_type_dist.values())
        if total_typed > 0:
            conflict = sum(
                v for k, v in event_type_dist.items()
                if "Conflict" in k
            )
            coop = total_typed - conflict
            conflict_pct = 100 * conflict / total_typed
            coop_pct = 100 * coop / total_typed
            top_type = max(event_type_dist, key=event_type_dist.get)

            if conflict_pct >= 60:
                s4_pool = _S4_CONFLICT
                sentences.append(_pick(s4_pool, seed + 3).format(
                    conflict_pct=conflict_pct, top_type=top_type,
                ))
            elif coop_pct >= 60:
                s4_pool = _S4_COOP
                sentences.append(_pick(s4_pool, seed + 3).format(
                    coop_pct=coop_pct, top_type=top_type,
                ))
            else:
                sentences.append(_pick(_S4_MIXED, seed + 3).format(
                    conflict_pct=conflict_pct, coop_pct=coop_pct,
                    top_type=top_type,
                ))

    
    if rag_chunks:
        domains = list({
            c.get("domain") or c.get("source_url", "")[:30]
            for c in rag_chunks[:4] if c.get("domain") or c.get("source_url")
        })
        domain_str = ", ".join(d for d in domains[:3] if d)
        if domain_str:
            sentences.append(
                _pick(_S5_WITH_EVIDENCE, seed + 4).format(domains=domain_str)
            )
    else:
        sentences.append(_S5_NO_EVIDENCE[0])

   
    if tone_delta < -1.0:
        sentences.append(_pick(_S6_ESCALATION, seed + 5))
    elif tone_delta > 1.0:
        sentences.append(_pick(_S6_DEESCALATION, seed + 5))
    else:
        sentences.append(_pick(_S6_GENERIC, seed + 5).format(country=country))

    summary = " ".join(sentences)

    
    if use_ollama and is_ollama_available(ollama_model):
        summary = refine_with_ollama(summary, preferred_model=ollama_model)

    return summary


def spike_info_from_row(row, df=None, burst_df=None) -> Dict:
    """
    Build the spike_info dict from a burst_summary row.

    Parameters
    ----------
    row      : A row from get_burst_summary() output
    df       : Full events DataFrame (optional, for tone delta computation)
    burst_df : Full burst DataFrame (optional)

    Returns
    -------
    Dict suitable for passing to summarize_spike()
    """
    import pandas as pd
    import numpy as np

    date_str = row["day"].strftime("%Y-%m-%d") if hasattr(row["day"], "strftime") else str(row["day"])
    country  = str(row["country"])
    count    = int(row["event_count"])
    baseline = float(row.get("rolling_mean", 0))
    z        = float(row.get("z_score", 0))

    
    tone_delta = 0.0
    if df is not None and "AvgTone" in df.columns:
        try:
            day_ts = pd.Timestamp(date_str)
            day_mask = (df["day"] == day_ts) & (df["country"] == country)
            day_tone = df[day_mask]["AvgTone"].mean()

           
            base_start = day_ts - pd.Timedelta(days=7)
            base_mask = (
                (df["day"] >= base_start) &
                (df["day"] < day_ts) &
                (df["country"] == country)
            )
            base_tone = df[base_mask]["AvgTone"].mean()

            if pd.notna(day_tone) and pd.notna(base_tone):
                tone_delta = round(float(day_tone - base_tone), 2)
        except Exception:
            pass

    return {
        "country"    : country,
        "date_str"   : date_str,
        "event_count": count,
        "baseline"   : baseline,
        "z_score"    : z,
        "tone_delta" : tone_delta,
    }


def get_event_type_dist(df, date_str: str, country: str) -> Dict:
    """
    Return event type distribution for a burst day.

    Returns
    -------
    Dict like {"Verbal Conflict": 320, "Material Cooperation": 150}
    """
    import pandas as pd

    day_ts = pd.Timestamp(date_str)
    mask = (df["day"] == day_ts) & (df["country"] == country)
    day_df = df[mask]

    if day_df.empty or "QuadLabel" not in day_df.columns:
        return {}

    return day_df["QuadLabel"].value_counts().to_dict()
