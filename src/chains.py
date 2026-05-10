import numpy as np
import pandas as pd
from datetime import timedelta

from .config import CHAIN_MAX_POSSIBLE


FEATURE_NAMES = [
    "same_country",
    "same_event_root",
    "same_quad_class",
    "same_location",
    "actor_exact_match",
    "actor_country_match",
    "actor_country_interact",
    "tone_distance",
    "goldstein_distance",
    "temporal_decay",
    "is_precursor",
    "is_followup",
    "candidate_strength",
    "anchor_strength",
    "mention_ratio",
    "cross_country",
]

NUM_FEATURES = len(FEATURE_NAMES)


HEURISTIC_WEIGHTS = np.array([
    2.5,    
    1.5,    
    1.5,    
    1.0,    
    3.0,    
    1.5,    
    2.0,   
    -1.0,   
    -1.0,   
    3.0,    
    0.3,    
    0.3,    
    1.0,    
    0.0,   
    0.5,    
    1.5,    
], dtype=np.float64)

TEMPORAL_TAU = 3.0  


def _kendall_tau(x, y):
    """
    Compute Kendall's tau-b and a two-sided p-value approximation.
    Pure numpy — no scipy dependency.
    Returns (tau, p_value).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 0.0, 1.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    n_pairs = n * (n - 1) / 2
    denom = np.sqrt((n_pairs - ties_x) * (n_pairs - ties_y))
    if denom == 0:
        return 0.0, 1.0

    tau = (concordant - discordant) / denom

  
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    s = concordant - discordant
    if var_s == 0:
        return tau, 1.0
    z = s / np.sqrt(var_s)
   
    p_value = 2.0 * _norm_sf(abs(z))
    return float(tau), float(p_value)


def _norm_sf(z):
    """Survival function (1-CDF) of standard normal, Abramowitz & Stegun approx."""
    if z < 0:
        return 1.0 - _norm_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * np.exp(-z * z / 2.0) * t * (
        0.319381530
        + t * (-0.356563782
               + t * (1.781477937
                      + t * (-1.821255978
                             + t * 1.330274429)))
    )
    return max(0.0, min(1.0, p))



def find_chain(
    df: pd.DataFrame,
    event_id: int,
    window_days: int = 7,
    top_n: int = 5,
    country_filter: str | None = None,
    allow_cross_country: bool = False,
    model=None,
    scaler=None,
) -> dict:
    """
    Build a scored event chain around an anchor event.

    Parameters
    ----------
    df               : cleaned events dataframe
    event_id         : GLOBALEVENTID of the anchor
    window_days      : look +/-N days around anchor
    top_n            : return top-N before and after
    country_filter   : restrict to one country (None = anchor's country)
    allow_cross_country : if True, search all countries
    model            : trained model dict (optional, from chain_model.py)
    scaler           : fitted scaler dict (optional, from chain_model.py)

    Returns
    -------
    dict with keys: previous, selected, next, explanation,
                    pattern, pattern_detail, narrative
    """
    anchor = df[df["GLOBALEVENTID"] == event_id]
    if anchor.empty:
        return _empty_chain()

    anchor_row = anchor.iloc[0]
    anchor_date = anchor_row["event_date"]

    if pd.isna(anchor_date):
        return _empty_chain()

   
    start = anchor_date - timedelta(days=window_days)
    end = anchor_date + timedelta(days=window_days)

    mask = (
        (df["event_date"] >= start)
        & (df["event_date"] <= end)
        & (df["GLOBALEVENTID"] != event_id)
    )

    if not allow_cross_country:
        filter_country = country_filter or anchor_row.get("country")
        if filter_country:
            mask &= df["country"] == filter_country

    candidates = df[mask].copy()

    if candidates.empty:
        return _isolated_chain(anchor_row)

   
    F = compute_features_batch(anchor_row, candidates)

    
    if model is not None and scaler is not None:
        scores = _predict_with_model(F, model, scaler)
    else:
        scores = F @ HEURISTIC_WEIGHTS

    candidates["chain_score"] = np.round(scores, 4)
    candidates["score_reasons"] = _build_reason_strings(F)

   
    before = (
        candidates[candidates["event_date"] < anchor_date]
        .nlargest(top_n, "chain_score")
        .sort_values("event_date")
    )
    after = (
        candidates[candidates["event_date"] > anchor_date]
        .nlargest(top_n, "chain_score")
        .sort_values("event_date")
    )

   
    same_day = candidates[candidates["event_date"] == anchor_date]
    if not same_day.empty:
        same_sorted = same_day.nlargest(top_n, "chain_score")
        half = max(len(same_sorted) // 2, 1)
        if len(before) < top_n:
            need = min(top_n - len(before), half)
            before = pd.concat([before, same_sorted.head(need)]).nlargest(
                top_n, "chain_score"
            ).sort_values("event_date")
        if len(after) < top_n:
            need = min(top_n - len(after), len(same_sorted) - half)
            if need > 0:
                after = pd.concat([after, same_sorted.tail(need)]).nlargest(
                    top_n, "chain_score"
                ).sort_values("event_date")

    cols = _result_columns(df)
    prev_list = before[cols].to_dict("records")
    next_list = after[cols].to_dict("records")

    
    all_events = prev_list + [_format_anchor(anchor_row)] + next_list
    pattern_detail = classify_chain_pattern(all_events)
    narrative = generate_narrative(
        anchor_row, prev_list, next_list, pattern_detail,
        country_filter or anchor_row.get("country", ""),
    )

    explanation = (
        "Events scored using 16-feature vector: country, actor (exact + fuzzy), "
        "event type, quad class, location, tone/Goldstein distance, "
        "temporal decay (τ=3d), direction, importance, cross-country link. "
        "Scored via trained model or heuristic weights."
    )

    return {
        "previous": prev_list,
        "selected": _format_anchor(anchor_row),
        "next": next_list,
        "explanation": explanation,
        "pattern": pattern_detail["pattern"],
        "pattern_detail": pattern_detail,
        "narrative": narrative,
    }


def _predict_with_model(F, model, scaler):
    """Score using our lightweight logistic regression."""
    mean = np.array(scaler["mean"])
    std = np.array(scaler["std"])
    std = np.where(std < 1e-9, 1.0, std)
    X_scaled = (F - mean) / std
    w = np.array(model["coef"])
    b = float(model["intercept"])
    logits = X_scaled @ w + b
    return 1.0 / (1.0 + np.exp(-logits))




def compute_features(anchor: pd.Series, candidate: pd.Series) -> np.ndarray:
    """Compute 16-dim feature vector for a single (anchor, candidate) pair."""
    f = np.zeros(NUM_FEATURES, dtype=np.float64)

    f[0] = float(anchor.get("country") == candidate.get("country"))
    f[1] = float(str(anchor.get("EventRootCode", "")) == str(candidate.get("EventRootCode", "")))
    f[2] = float(anchor.get("QuadClass") == candidate.get("QuadClass"))

    a_loc = _safe_str(anchor.get("ActionGeo_FullName"))
    c_loc = _safe_str(candidate.get("ActionGeo_FullName"))
    f[3] = float(a_loc != "" and a_loc == c_loc)

    anchor_actors = _actor_set(anchor)
    cand_actors = _actor_set(candidate)
    f[4] = float(bool(anchor_actors & cand_actors))
    f[5] = float(_actor_country_match(anchor, candidate))
    f[6] = f[4] * f[0]

    a_tone = float(anchor.get("AvgTone", 0) or 0)
    c_tone = float(candidate.get("AvgTone", 0) or 0)
    f[7] = min(abs(a_tone - c_tone) / 20.0, 1.0)

    a_gold = float(anchor.get("GoldsteinScale", 0) or 0)
    c_gold = float(candidate.get("GoldsteinScale", 0) or 0)
    f[8] = min(abs(a_gold - c_gold) / 20.0, 1.0)

    delta = abs((candidate["event_date"] - anchor["event_date"]).total_seconds()) / 86400
    f[9] = np.exp(-delta / TEMPORAL_TAU)
    f[10] = float(candidate["event_date"] < anchor["event_date"])
    f[11] = float(candidate["event_date"] > anchor["event_date"])

    f[12] = min(float(candidate.get("event_strength", 0) or 0) / 10.0, 1.0)
    f[13] = min(float(anchor.get("event_strength", 0) or 0) / 10.0, 1.0)

    a_ment = max(float(anchor.get("NumMentions", 1) or 1), 1)
    c_ment = max(float(candidate.get("NumMentions", 1) or 1), 1)
    f[14] = min(c_ment / a_ment, 5.0) / 5.0

    f[15] = float(anchor.get("country") != candidate.get("country") and f[4] > 0)

    return f


def compute_features_batch(
    anchor: pd.Series, candidates: pd.DataFrame,
) -> np.ndarray:
    """Vectorised feature computation. Returns shape (n_candidates, 16)."""
    n = len(candidates)
    F = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    a_country = anchor.get("country")
    a_root = str(anchor.get("EventRootCode", ""))
    a_quad = anchor.get("QuadClass")
    a_loc = _safe_str(anchor.get("ActionGeo_FullName"))
    a_tone = float(anchor.get("AvgTone", 0) or 0)
    a_gold = float(anchor.get("GoldsteinScale", 0) or 0)
    a_strength = float(anchor.get("event_strength", 0) or 0)
    a_mentions = max(float(anchor.get("NumMentions", 1) or 1), 1.0)
    anchor_date = anchor["event_date"]

    anchor_actors = _actor_set(anchor)
    a_cc1 = _safe_str(anchor.get("Actor1CountryCode"))
    a_cc2 = _safe_str(anchor.get("Actor2CountryCode"))

    F[:, 0] = (candidates["country"].values == a_country).astype(np.float64)
    F[:, 1] = (candidates["EventRootCode"].astype(str).values == a_root).astype(np.float64)
    F[:, 2] = (candidates["QuadClass"].values == a_quad).astype(np.float64)

    if a_loc:
        F[:, 3] = (candidates["ActionGeo_FullName"].fillna("").values == a_loc).astype(np.float64)

    if anchor_actors:
        c_a1 = candidates["actor1_clean"].values
        c_a2 = candidates["actor2_clean"].values
        match = np.zeros(n, dtype=bool)
        for actor in anchor_actors:
            match |= (c_a1 == actor) | (c_a2 == actor)
        F[:, 4] = match.astype(np.float64)

    cc1_vals = candidates["Actor1CountryCode"].fillna("").values
    cc2_vals = candidates["Actor2CountryCode"].fillna("").values
    fuzzy = np.zeros(n, dtype=bool)
    for code in (a_cc1, a_cc2):
        if code and code not in ("", "UNK"):
            fuzzy |= (cc1_vals == code) | (cc2_vals == code)
    F[:, 5] = fuzzy.astype(np.float64)

    F[:, 6] = F[:, 4] * F[:, 0]

    c_tones = candidates["AvgTone"].fillna(0).values.astype(np.float64)
    F[:, 7] = np.minimum(np.abs(c_tones - a_tone) / 20.0, 1.0)

    c_golds = candidates["GoldsteinScale"].fillna(0).values.astype(np.float64)
    F[:, 8] = np.minimum(np.abs(c_golds - a_gold) / 20.0, 1.0)

    delta_sec = (candidates["event_date"] - anchor_date).dt.total_seconds().values
    abs_days = np.abs(delta_sec) / 86400.0
    F[:, 9] = np.exp(-abs_days / TEMPORAL_TAU)
    F[:, 10] = (delta_sec < 0).astype(np.float64)
    F[:, 11] = (delta_sec > 0).astype(np.float64)

    F[:, 12] = np.minimum(
        candidates["event_strength"].fillna(0).values.astype(np.float64) / 10.0, 1.0,
    )
    F[:, 13] = min(a_strength / 10.0, 1.0)

    c_mentions = np.maximum(candidates["NumMentions"].fillna(1).values.astype(np.float64), 1.0)
    F[:, 14] = np.minimum(c_mentions / a_mentions, 5.0) / 5.0

    F[:, 15] = ((F[:, 0] == 0) & (F[:, 4] > 0)).astype(np.float64)

    return F



def classify_chain_pattern(chain_events: list) -> dict:
    """Classify chain pattern using Mann-Kendall trend test on GoldsteinScale."""
    if len(chain_events) < 2:
        return {"pattern": "Isolated", "tau": 0.0, "p_value": 1.0,
                "confidence": "Low", "goldstein_trajectory": []}

    sorted_events = sorted(
        chain_events,
        key=lambda e: str(e.get("event_date", e.get("day", ""))),
    )
    goldstein = []
    for e in sorted_events:
        g = e.get("GoldsteinScale")
        if g is not None:
            try:
                gf = float(g)
                if not np.isnan(gf):
                    goldstein.append(gf)
            except (ValueError, TypeError):
                pass

    if len(goldstein) < 3:
        return _simple_pattern_fallback(chain_events, goldstein)

    time_idx = list(range(len(goldstein)))
    tau, p_value = _kendall_tau(time_idx, goldstein)

    if p_value < 0.10 and abs(tau) > 0.3:
        if tau < -0.3:
            pattern = "Escalation"
            confidence = "High" if p_value < 0.05 else "Moderate"
        else:
            pattern = "De-escalation"
            confidence = "High" if p_value < 0.05 else "Moderate"
    elif p_value >= 0.10:
        gold_std = float(np.std(goldstein))
        if gold_std > 4.0:
            pattern = "Mixed"
            confidence = "Moderate"
        else:
            pattern = "Persistence"
            confidence = "Low"
    else:
        pattern = "Mixed"
        confidence = "Low"

    return {
        "pattern": pattern, "tau": round(tau, 3), "p_value": round(p_value, 4),
        "confidence": confidence, "goldstein_trajectory": goldstein,
    }


def _simple_pattern_fallback(chain_events, goldstein):
    quads = []
    for e in chain_events:
        q = e.get("QuadClass")
        if q is not None:
            try:
                quads.append(int(float(q)))
            except (ValueError, TypeError):
                pass
    if not quads:
        return {"pattern": "Isolated", "tau": 0.0, "p_value": 1.0,
                "confidence": "Low", "goldstein_trajectory": goldstein}
    cr = sum(1 for q in quads if q in (3, 4)) / len(quads)
    pattern = "Persistence" if (cr > 0.7 or cr < 0.3) else "Mixed"
    return {"pattern": pattern, "tau": 0.0, "p_value": 1.0,
            "confidence": "Low", "goldstein_trajectory": goldstein}



def generate_narrative(anchor_row, prev_list, next_list, pattern_detail, country):
    total = len(prev_list) + len(next_list)
    if total == 0:
        return "This event appears isolated — no strongly related events found nearby."

    anchor_type = str(anchor_row.get("QuadLabel", "event")).lower()
    anchor_root = str(anchor_row.get("EventRootLabel", "")).lower()
    a1 = anchor_row.get("actor1_clean", "Unknown")
    a2 = anchor_row.get("actor2_clean", "Unknown")
    actors = f"{a1} and {a2}" if a1 != "Unknown" and a2 != "Unknown" else (a1 if a1 != "Unknown" else "unidentified actors")

    parts = []
    root_note = f" ({anchor_root})" if anchor_root and anchor_root not in ("nan", "") else ""
    country_note = f" in {country}" if country else ""
    parts.append(f"This {anchor_type} event{root_note} involving {actors}{country_note} is connected to {total} related events.")

    pat = pattern_detail["pattern"]
    tau = pattern_detail.get("tau", 0)
    p_val = pattern_detail.get("p_value", 1)
    conf = pattern_detail.get("confidence", "Low")

    desc = {
        "Escalation": f"The chain shows an escalation pattern (Kendall τ={tau:.2f}, p={p_val:.3f}, {conf.lower()} confidence) — conflict intensity increases over the sequence.",
        "De-escalation": f"The chain shows a de-escalation pattern (Kendall τ={tau:.2f}, p={p_val:.3f}, {conf.lower()} confidence) — tensions ease over the sequence.",
        "Persistence": f"No significant directional trend detected — the event type persists with consistent intensity (τ={tau:.2f}, p={p_val:.3f}).",
        "Mixed": f"The chain shows high variability with no clear direction (τ={tau:.2f}, p={p_val:.3f}). Events alternate between conflict and cooperation.",
    }
    if pat in desc:
        parts.append(desc[pat])

    traj = pattern_detail.get("goldstein_trajectory", [])
    if len(traj) >= 3:
        g_start = np.mean(traj[:2])
        g_end = np.mean(traj[-2:])
        if g_end < g_start - 2:
            parts.append(f"Conflict intensity increased (Goldstein: {g_start:.1f} → {g_end:.1f}).")
        elif g_end > g_start + 2:
            parts.append(f"Tensions eased over time (Goldstein: {g_start:.1f} → {g_end:.1f}).")

    return " ".join(parts)



def _safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _actor_set(row) -> set:
    a1 = _safe_str(row.get("actor1_clean"))
    a2 = _safe_str(row.get("actor2_clean"))
    return {a1, a2} - {"Unknown", ""}


def _actor_country_match(anchor, candidate) -> bool:
    codes = set()
    for field in ("Actor1CountryCode", "Actor2CountryCode"):
        v = _safe_str(anchor.get(field))
        if v and v != "UNK":
            codes.add(v)
    if not codes:
        return False
    for field in ("Actor1CountryCode", "Actor2CountryCode"):
        v = _safe_str(candidate.get(field))
        if v in codes:
            return True
    return False


_BINARY_REASON_INDICES = [
    (0, "country"), (1, "event_family"), (2, "quad_class"),
    (3, "location"), (4, "actor"), (5, "actor_fuzzy"), (15, "cross_country"),
]


def _build_reason_strings(F: np.ndarray) -> list[str]:
    reasons = []
    for i in range(F.shape[0]):
        parts = []
        for idx, label in _BINARY_REASON_INDICES:
            if F[i, idx] > 0:
                parts.append(label)
        if F[i, 12] > 0.5:
            parts.append("importance")
        reasons.append(", ".join(parts))
    return reasons


def _format_anchor(row) -> dict:
    keys = [
        "GLOBALEVENTID", "event_date", "actor1_clean", "actor2_clean",
        "country", "EventRootCode", "EventRootLabel", "QuadLabel", "EventType",
        "AvgTone", "GoldsteinScale", "ActionGeo_FullName", "SOURCEURL",
        "event_label", "event_strength", "QuadClass",
    ]
    result = {}
    for k in keys:
        val = row.get(k) if hasattr(row, "get") else row[k] if k in row.index else None
        if pd.isna(val):
            val = ""
        result[k] = val
    return result


def _result_columns(df: pd.DataFrame) -> list:
    desired = [
        "GLOBALEVENTID", "event_date", "actor1_clean", "actor2_clean",
        "country", "EventRootCode", "EventRootLabel", "QuadLabel", "EventType",
        "AvgTone", "GoldsteinScale", "ActionGeo_FullName", "SOURCEURL",
        "chain_score", "score_reasons", "event_label", "event_strength", "QuadClass",
    ]
    return [c for c in desired if c in df.columns or c in ("chain_score", "score_reasons")]


def _empty_chain():
    d = {"pattern": "Unknown", "tau": 0, "p_value": 1, "confidence": "Low", "goldstein_trajectory": []}
    return {"previous": [], "selected": None, "next": [], "explanation": "",
            "pattern": "Unknown", "pattern_detail": d, "narrative": ""}


def _isolated_chain(anchor_row):
    d = {"pattern": "Isolated", "tau": 0, "p_value": 1, "confidence": "Low", "goldstein_trajectory": []}
    return {"previous": [], "selected": _format_anchor(anchor_row), "next": [],
            "explanation": "No candidate events found in the time window.",
            "pattern": "Isolated", "pattern_detail": d,
            "narrative": "This event appears isolated — no related events found nearby."}
