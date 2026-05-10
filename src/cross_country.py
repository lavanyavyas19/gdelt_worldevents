
import numpy as np
import pandas as pd
from datetime import timedelta
from itertools import combinations

from .chains import (
    compute_features_batch, HEURISTIC_WEIGHTS, FEATURE_NAMES,
    _format_anchor, _result_columns,
)


def _pearsonr(x, y):
    """
    Compute Pearson correlation coefficient and two-sided p-value.
    Pure numpy — no scipy.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mx = x.mean()
    my = y.mean()
    xm = x - mx
    ym = y - my

    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))

    if r_den < 1e-12:
        return 0.0, 1.0

    r = r_num / r_den
    r = max(-1.0, min(1.0, r))


    if abs(r) >= 1.0:
        return float(r), 0.0

    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))

    p_value = 2.0 * _t_sf(abs(t_stat), n - 2)
    return float(r), float(p_value)


def _t_sf(t, df):
    """Approximate survival function of t-distribution using normal approx."""
    
    if df <= 0:
        return 0.5
    if df > 30:
        z = t
    else:
        
        z = t * np.sqrt(max(df - 2, 0.5) / max(df, 1))

    
    return _norm_sf(z)


def _norm_sf(z):
    """Survival function (1-CDF) of standard normal."""
    if z < 0:
        return 1.0 - _norm_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327
    p = d * np.exp(-z * z / 2.0) * t * (
        0.319381530
        + t * (-0.356563782
               + t * (1.781477937
                      + t * (-1.821255978
                             + t * 1.330274429)))
    )
    return max(0.0, min(1.0, p))




def compute_lag_correlation(
    burst_df: pd.DataFrame,
    country_a: str,
    country_b: str,
    max_lag: int = 7,
    metric: str = "event_count",
) -> dict:
    """
    Compute lead-lag correlation between two countries' daily event series.

    Positive best_lag means country_a leads country_b by that many days.
    Negative best_lag means country_b leads country_a.
    """
    a_df = (
        burst_df[burst_df["country"] == country_a]
        .set_index("day")[metric]
        .sort_index()
    )
    b_df = (
        burst_df[burst_df["country"] == country_b]
        .set_index("day")[metric]
        .sort_index()
    )

    common = a_df.index.intersection(b_df.index)
    if len(common) < 14:
        return {
            "country_a": country_a, "country_b": country_b,
            "error": f"Not enough overlapping dates ({len(common)})",
        }

    a = a_df.loc[common].values.astype(float)
    b = b_df.loc[common].values.astype(float)
    n = len(a)

    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a_s, b_s = a[:-lag], b[lag:]
        elif lag < 0:
            a_s, b_s = a[-lag:], b[:lag]
        else:
            a_s, b_s = a, b

        ml = min(len(a_s), len(b_s))
        if ml < 10:
            continue

        if np.std(a_s[:ml]) < 1e-9 or np.std(b_s[:ml]) < 1e-9:
            results.append({"lag": lag, "correlation": 0.0, "p_value": 1.0})
            continue

        corr, p_val = _pearsonr(a_s[:ml], b_s[:ml])
        results.append({
            "lag": lag,
            "correlation": round(corr, 4),
            "p_value": round(p_val, 4),
        })

    if not results:
        return {
            "country_a": country_a, "country_b": country_b,
            "error": "No valid lag computations",
        }

    significant = [r for r in results if r["p_value"] < 0.05]
    pool = significant if significant else results
    best = max(pool, key=lambda r: abs(r["correlation"]))

    return {
        "country_a": country_a,
        "country_b": country_b,
        "lag_results": results,
        "best_lag": best["lag"],
        "best_correlation": best["correlation"],
        "best_p_value": best["p_value"],
        "n_common_days": n,
        "interpretation": _interpret_lag(country_a, country_b, best),
    }


def compute_all_pairs(
    burst_df: pd.DataFrame,
    countries: list[str],
    max_lag: int = 7,
) -> list[dict]:
    """Compute lead-lag correlation for all country pairs."""
    results = []
    for a, b in combinations(countries, 2):
        results.append(compute_lag_correlation(burst_df, a, b, max_lag))
    return results


def _interpret_lag(country_a, country_b, best):
    lag = best["lag"]
    corr = best["correlation"]
    p = best["p_value"]

    if abs(corr) < 0.15:
        return f"No meaningful correlation between {country_a} and {country_b}."

    sig = " (statistically significant)" if p < 0.05 else ""
    direction = "positive" if corr > 0 else "inverse"

    if lag > 0:
        return f"Spikes in {country_a} precede spikes in {country_b} by ~{lag}d ({direction} r={corr:.3f}){sig}."
    elif lag < 0:
        return f"Spikes in {country_b} precede spikes in {country_a} by ~{abs(lag)}d ({direction} r={corr:.3f}){sig}."
    else:
        return f"Spikes in {country_a} and {country_b} co-occur ({direction} r={corr:.3f}){sig}."




def cross_country_spike_overlap(burst_df: pd.DataFrame) -> pd.DataFrame:
    """Find days where 2+ countries spike simultaneously."""
    burst_days = burst_df[burst_df["is_burst"]][
        ["day", "country", "z_score", "event_count"]
    ].copy()

    if burst_days.empty:
        return pd.DataFrame(columns=["day", "country", "z_score", "event_count", "n_countries"])

    day_counts = burst_days.groupby("day")["country"].nunique().reset_index(name="n_countries")
    multi = day_counts[day_counts["n_countries"] >= 2]

    if multi.empty:
        return pd.DataFrame(columns=["day", "country", "z_score", "event_count", "n_countries"])

    return burst_days.merge(multi[["day", "n_countries"]], on="day").sort_values(["day", "country"]).reset_index(drop=True)



def find_cross_country_links(
    df: pd.DataFrame,
    event_id: int,
    window_days: int = 5,
    top_n: int = 5,
) -> dict:
    """
    Find events in OTHER countries related to the anchor event.
    Requires shared actor code/country code to qualify.
    """
    anchor = df[df["GLOBALEVENTID"] == event_id]
    if anchor.empty:
        return {"anchor": None, "transnational_links": [], "countries_involved": []}

    anchor_row = anchor.iloc[0]
    anchor_date = anchor_row["event_date"]
    anchor_country = anchor_row["country"]

    if pd.isna(anchor_date):
        return {"anchor": None, "transnational_links": [], "countries_involved": []}

    start = anchor_date - timedelta(days=window_days)
    end = anchor_date + timedelta(days=window_days)

    candidates = df[
        (df["event_date"] >= start) & (df["event_date"] <= end)
        & (df["country"] != anchor_country) & (df["GLOBALEVENTID"] != event_id)
    ].copy()

    if candidates.empty:
        return {
            "anchor": _format_anchor(anchor_row),
            "transnational_links": [],
            "countries_involved": [],
        }

    F = compute_features_batch(anchor_row, candidates)

    # Cross-country weights
    cross_w = HEURISTIC_WEIGHTS.copy()
    cross_w[FEATURE_NAMES.index("same_country")] = 0.0
    cross_w[FEATURE_NAMES.index("same_location")] = 0.0
    cross_w[FEATURE_NAMES.index("actor_exact_match")] = 5.0
    cross_w[FEATURE_NAMES.index("actor_country_match")] = 3.0
    cross_w[FEATURE_NAMES.index("same_event_root")] = 2.5
    cross_w[FEATURE_NAMES.index("temporal_decay")] = 4.0
    cross_w[FEATURE_NAMES.index("cross_country")] = 3.0

    scores = F @ cross_w
    candidates["chain_score"] = np.round(scores, 4)

    # Require actor signal
    has_actor_link = (F[:, 4] > 0) | (F[:, 5] > 0)
    qualified = candidates[has_actor_link].copy()

    if qualified.empty:
        return {
            "anchor": _format_anchor(anchor_row),
            "transnational_links": [],
            "countries_involved": [],
        }

    top = qualified.nlargest(top_n, "chain_score")
    cols = _result_columns(df)

    return {
        "anchor": _format_anchor(anchor_row),
        "transnational_links": top[cols].to_dict("records"),
        "countries_involved": sorted(top["country"].unique().tolist()),
    }
