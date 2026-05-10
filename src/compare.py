from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd




def get_historical_spikes(
    burst_df: pd.DataFrame,
    country: str,
    current_date: str,
    n: int = 5,
) -> List[Dict]:
    """
    Return up to `n` most recent burst days before the current date
    for the given country.

    Parameters
    ----------
    burst_df     : Full burst DataFrame (from detect_bursts or load_bursts)
    country      : Country name, e.g. "Iran"
    current_date : "YYYY-MM-DD" — current spike to exclude
    n            : Max number of past spikes to retrieve

    Returns
    -------
    List of dicts: [{"day": Timestamp, "event_count": int,
                     "z_score": float, "rolling_mean": float}, ...]
    Sorted by date descending (most recent first).
    """
    current_ts = pd.Timestamp(current_date)

    past = burst_df[
        (burst_df["country"] == country) &
        (burst_df["is_burst"]) &
        (burst_df["day"] < current_ts)
    ].copy()

    if past.empty:
        return []

    past = past.sort_values("day", ascending=False).head(n)
    return past[["day", "event_count", "z_score", "rolling_mean"]].to_dict("records")



def compute_spike_stats(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    date_str: str,
    country: str,
    window_days: int = 3,
) -> Dict:
    """
    Compute rich statistics for a single burst day.

    Parameters
    ----------
    df          : Full events DataFrame
    burst_df    : Burst detection results
    date_str    : "YYYY-MM-DD" burst date
    country     : Country name
    window_days : Days on either side for tone trajectory

    Returns
    -------
    Dict with:
        date_str, country, event_count, z_score, baseline,
        avg_tone, tone_trajectory (list of daily tones),
        conflict_pct, coop_pct,
        top_quad (str), consecutive_burst_days (int)
    """
    day_ts = pd.Timestamp(date_str)

    b_row = burst_df[
        (burst_df["day"] == day_ts) &
        (burst_df["country"] == country)
    ]
    event_count = int(b_row["event_count"].iloc[0]) if not b_row.empty else 0
    z_score     = float(b_row["z_score"].iloc[0]) if not b_row.empty else 0.0
    baseline    = float(b_row["rolling_mean"].iloc[0]) if not b_row.empty else 0.0

   
    day_mask = (df["day"] == day_ts) & (df["country"] == country)
    day_df   = df[day_mask].copy()

  
    avg_tone = float(day_df["AvgTone"].mean()) if not day_df.empty and "AvgTone" in day_df.columns else 0.0

    
    conflict_pct = coop_pct = 0.0
    top_quad = "Unknown"
    if not day_df.empty and "QuadClass" in day_df.columns:
        n_total  = len(day_df)
        conflict = int((day_df["QuadClass"] >= 3).sum())
        conflict_pct = 100 * conflict / max(n_total, 1)
        coop_pct     = 100 - conflict_pct
        if "QuadLabel" in day_df.columns:
            top_quad = day_df["QuadLabel"].value_counts().idxmax()

   
    start_ts = day_ts - pd.Timedelta(days=window_days)
    end_ts   = day_ts + pd.Timedelta(days=window_days)
    traj_mask = (
        (df["day"] >= start_ts) &
        (df["day"] <= end_ts) &
        (df["country"] == country) &
        df["AvgTone"].notna()
    )
    traj_df = df[traj_mask].groupby("day")["AvgTone"].mean().sort_index()
    tone_trajectory = [round(float(v), 2) for v in traj_df.values]

   
    country_burst = burst_df[
        (burst_df["country"] == country) & (burst_df["is_burst"])
    ].sort_values("day")
    burst_days = set(country_burst["day"].dt.normalize())
    consecutive = 1
    check = day_ts - pd.Timedelta(days=1)
    while check in burst_days:
        consecutive += 1
        check -= pd.Timedelta(days=1)
    check = day_ts + pd.Timedelta(days=1)
    while check in burst_days:
        consecutive += 1
        check += pd.Timedelta(days=1)

    return {
        "date_str"             : date_str,
        "country"              : country,
        "event_count"          : event_count,
        "z_score"              : z_score,
        "baseline"             : baseline,
        "avg_tone"             : avg_tone,
        "tone_trajectory"      : tone_trajectory,
        "conflict_pct"         : conflict_pct,
        "coop_pct"             : coop_pct,
        "top_quad"             : top_quad,
        "consecutive_burst_days": consecutive,
    }


def compute_historical_stats_bulk(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    historical_spikes: List[Dict],
    country: str,
) -> List[Dict]:
    """
    Compute stats for each historical spike.

    Returns
    -------
    List[Dict] — one stats dict per historical spike.
    """
    stats = []
    for spike in historical_spikes:
        date_str = spike["day"].strftime("%Y-%m-%d") if hasattr(spike["day"], "strftime") else str(spike["day"])
        try:
            s = compute_spike_stats(df, burst_df, date_str, country)
            stats.append(s)
        except Exception:
            pass
    return stats




def compare_spikes(
    current: Dict,
    historical: List[Dict],
) -> Dict:
    """
    Compare current spike statistics against historical averages.

    Parameters
    ----------
    current    : Stats dict from compute_spike_stats() for the current spike
    historical : List of stats dicts for past spikes

    Returns
    -------
    Dict with comparison metrics:
        count_ratio, z_ratio, tone_diff, conflict_pct_diff,
        duration_ratio, escalation_direction,
        vs_each: List[Dict]  (per-spike comparison)
    """
    if not historical:
        return {
            "count_ratio": None, "z_ratio": None, "tone_diff": None,
            "conflict_pct_diff": None, "duration_ratio": None,
            "escalation_direction": "Unknown", "vs_each": [],
            "has_history": False,
        }

    hist_counts    = [h["event_count"]   for h in historical]
    hist_z         = [h["z_score"]       for h in historical]
    hist_tone      = [h["avg_tone"]      for h in historical]
    hist_conflict  = [h["conflict_pct"]  for h in historical]
    hist_duration  = [h["consecutive_burst_days"] for h in historical]

    avg_count    = np.mean(hist_counts)
    avg_z        = np.mean(hist_z)
    avg_tone     = np.mean(hist_tone)
    avg_conflict = np.mean(hist_conflict)
    avg_duration = np.mean(hist_duration)

    count_ratio    = current["event_count"] / max(avg_count, 1)
    z_ratio        = current["z_score"] / max(avg_z, 0.01)
    tone_diff      = current["avg_tone"] - avg_tone
    conflict_diff  = current["conflict_pct"] - avg_conflict
    duration_ratio = current["consecutive_burst_days"] / max(avg_duration, 1)

 
    traj = current.get("tone_trajectory", [])
    if len(traj) >= 3:
        first_half = np.mean(traj[:len(traj)//2])
        second_half= np.mean(traj[len(traj)//2:])
        escalation_direction = "Escalating" if second_half < first_half - 0.5 else (
            "De-escalating" if second_half > first_half + 0.5 else "Stable"
        )
    else:
        escalation_direction = "Insufficient data"

  
    vs_each = []
    for h in historical:
        h_date = h["date_str"] if "date_str" in h else str(h.get("day", "?"))
        vs_each.append({
            "date"         : h_date,
            "count_ratio"  : round(current["event_count"] / max(h["event_count"], 1), 2),
            "z_diff"       : round(current["z_score"] - h["z_score"], 2),
            "tone_diff"    : round(current["avg_tone"] - h["avg_tone"], 2),
            "conflict_diff": round(current["conflict_pct"] - h["conflict_pct"], 1),
        })

    return {
        "count_ratio"          : round(count_ratio, 2),
        "z_ratio"              : round(z_ratio, 2),
        "tone_diff"            : round(tone_diff, 2),
        "conflict_pct_diff"    : round(conflict_diff, 1),
        "duration_ratio"       : round(duration_ratio, 2),
        "escalation_direction" : escalation_direction,
        "avg_historical_count" : round(avg_count, 0),
        "avg_historical_z"     : round(avg_z, 2),
        "avg_historical_tone"  : round(avg_tone, 2),
        "n_historical"         : len(historical),
        "vs_each"              : vs_each,
        "has_history"          : True,
    }



def generate_comparison_narrative(comparison: Dict, country: str) -> str:
    """
    Generate a 3–5 sentence comparative narrative from comparison metrics.
    No LLM required.

    Parameters
    ----------
    comparison : Output of compare_spikes()
    country    : Country name for contextualisation

    Returns
    -------
    Multi-sentence analyst comparison narrative.
    """
    if not comparison.get("has_history"):
        return (
            f"No historical burst data is available for {country}, so a "
            "comparative assessment cannot be made. This appears to be the "
            "first detected spike in the current dataset."
        )

    count_ratio    = comparison["count_ratio"]
    z_ratio        = comparison["z_ratio"]
    tone_diff      = comparison["tone_diff"]
    conflict_diff  = comparison["conflict_pct_diff"]
    duration_ratio = comparison["duration_ratio"]
    direction      = comparison["escalation_direction"]
    n_hist         = comparison["n_historical"]
    avg_count      = comparison["avg_historical_count"]
    avg_z          = comparison["avg_historical_z"]

    sentences = []


    if count_ratio >= 1.5:
        sentences.append(
            f"This spike is {count_ratio:.1f}× larger than the average of the "
            f"previous {n_hist} burst{'' if n_hist == 1 else 's'} "
            f"(historical mean: {avg_count:,.0f} events), making it notably atypical."
        )
    elif count_ratio <= 0.7:
        sentences.append(
            f"This spike is smaller than average, with event counts "
            f"{(1-count_ratio)*100:.0f}% below the historical mean of {avg_count:,.0f} events."
        )
    else:
        sentences.append(
            f"Event volume is broadly consistent with historical spikes for "
            f"{country}, at {count_ratio:.1f}× the average of {avg_count:,.0f} events."
        )


    if z_ratio >= 1.5:
        sentences.append(
            f"The statistical intensity (z-score ratio: {z_ratio:.1f}×) significantly "
            f"exceeds the historical average of {avg_z:.1f}σ, indicating this is "
            "among the most statistically extreme events in the dataset."
        )
    elif z_ratio <= 0.7:
        sentences.append(
            f"Despite the absolute event volume, the z-score ({comparison.get('z_ratio',0):.1f}×) "
            f"is lower than historical averages ({avg_z:.1f}σ), suggesting the baseline "
            "has also elevated."
        )

  
    if tone_diff < -1.5:
        sentences.append(
            f"Tone is {abs(tone_diff):.1f} points more negative than the historical "
            "average for burst periods, suggesting an unusually adversarial or "
            "hostile environment compared to past spikes."
        )
    elif tone_diff > 1.5:
        sentences.append(
            f"Tone is {tone_diff:.1f} points more positive than historical burst "
            "averages, indicating a more cooperative character than previous spikes."
        )
    else:
        sentences.append(
            f"Tone (Δ{tone_diff:+.1f} vs. historical average) is broadly consistent "
            "with previous burst periods for this country."
        )

   
    if abs(conflict_diff) >= 10:
        direction_word = "higher" if conflict_diff > 0 else "lower"
        sentences.append(
            f"The share of conflict-type events is {abs(conflict_diff):.0f} percentage "
            f"points {direction_word} than the historical average for burst days, "
            f"which is analytically significant."
        )

  
    if direction in ("Escalating", "De-escalating"):
        sentences.append(
            f"The tone trajectory over the burst window shows a {direction.lower()} "
            "pattern, distinguishing this event from a simple one-day shock."
        )

    return " ".join(sentences)


def format_comparison_table(
    current: Dict,
    historical: List[Dict],
    comparison: Dict,
) -> pd.DataFrame:
    """
    Format a comparison table for display in Streamlit.

    Returns
    -------
    DataFrame with columns: Date, Events, Z-Score, Avg Tone, Conflict%, Duration
    with the current spike highlighted at the top.
    """
    rows = [{
        "Date"        : current["date_str"] + " ★",
        "Events"      : current["event_count"],
        "Z-Score (σ)" : round(current["z_score"], 2),
        "Avg Tone"    : round(current["avg_tone"], 2),
        "Conflict %"  : round(current["conflict_pct"], 1),
        "Burst Days"  : current["consecutive_burst_days"],
    }]

    for h in historical:
        h_date = h.get("date_str", str(h.get("day", "?")))
        rows.append({
            "Date"        : h_date,
            "Events"      : h["event_count"],
            "Z-Score (σ)" : round(h["z_score"], 2),
            "Avg Tone"    : round(h["avg_tone"], 2),
            "Conflict %"  : round(h["conflict_pct"], 1),
            "Burst Days"  : h["consecutive_burst_days"],
        })

    return pd.DataFrame(rows)
