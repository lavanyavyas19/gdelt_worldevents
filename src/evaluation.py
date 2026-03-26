"""
evaluation.py
-------------
Evaluation framework for the GDELT Event Intelligence system.

Provides:
  1. Retrieval metrics: Precision@k, Recall@k, nDCG@k, MRR, MAP
  2. Burst evaluation against ground truth
  3. Chain retrieval evaluation against ground truth
  4. Pattern classification evaluation (accuracy, F1, confusion matrix)
  5. Ablation study (disable feature groups, measure nDCG delta)
  6. Baseline comparison (random, recency-only)
  7. Data quality metrics
  8. Ground truth template generation
  9. Response time benchmarking

Ground truth JSON format:
{
    "burst_ground_truth": [
        {"day": "YYYY-MM-DD", "country": "...", "is_real_burst": true}
    ],
    "chain_ground_truth": [
        {"anchor_event_id": N, "relevant_event_ids": [N, ...], "expected_pattern": "..."}
    ]
}
"""

import time
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, List

from .chains import (
    find_chain, compute_features_batch, HEURISTIC_WEIGHTS,
    FEATURE_NAMES, NUM_FEATURES, classify_chain_pattern,
)


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for eid in top_k if eid in relevant_ids)
    return round(hits / len(top_k), 4)


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant items found in top-k."""
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    hits = sum(1 for eid in top_k if eid in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance)."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    dcg = sum(
        (1.0 if eid in relevant_ids else 0.0) / np.log2(i + 2)
        for i, eid in enumerate(top_k)
    )
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    return round(dcg / max(idcg, 1e-9), 4)


def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """1 / rank of first relevant result."""
    for i, eid in enumerate(retrieved_ids):
        if eid in relevant_ids:
            return round(1.0 / (i + 1), 4)
    return 0.0


def mean_average_precision(retrieved_ids: list, relevant_ids: set) -> float:
    """Mean of precision at each relevant rank position."""
    precisions = []
    hits = 0
    for i, eid in enumerate(retrieved_ids):
        if eid in relevant_ids:
            hits += 1
            precisions.append(hits / (i + 1))
    if not precisions:
        return 0.0
    return round(sum(precisions) / len(relevant_ids), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_data_quality(df: pd.DataFrame, report: dict) -> dict:
    return {
        "rows_raw": report.get("rows_raw", 0),
        "rows_final": report.get("rows_final", 0),
        "invalid_dates_dropped": report.get("invalid_dates_dropped", 0),
        "duplicates_removed": report.get("duplicates_removed", 0),
        "rows_outside_countries": report.get("rows_outside_countries", 0),
        "rows_outside_window": report.get("rows_outside_window", 0),
        "date_range": f"{report.get('date_min', '?')} → {report.get('date_max', '?')}",
        "countries": report.get("countries", {}),
        "unique_days": int(df["day"].nunique()) if "day" in df.columns else 0,
        "missing_actor_pct": round(
            float(df["has_missing_actor"].mean() * 100), 1
        ) if "has_missing_actor" in df.columns else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BURST EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_burst_sanity(burst_df: pd.DataFrame) -> dict:
    total_rows = len(burst_df)
    burst_rows = int(burst_df["is_burst"].sum())
    return {
        "total_country_days": total_rows,
        "calendar_days": int(burst_df["day"].nunique()),
        "burst_country_days": burst_rows,
        "burst_ratio": round(burst_rows / max(total_rows, 1), 4),
        "mean_z_on_burst": round(
            float(burst_df[burst_df["is_burst"]]["z_score"].mean()), 3
        ) if burst_rows > 0 else 0,
        "max_z_score": round(float(burst_df["z_score"].max()), 3),
        "bursts_per_country": (
            burst_df[burst_df["is_burst"]]
            .groupby("country")["is_burst"].sum().to_dict()
        ) if burst_rows > 0 else {},
    }


def evaluate_burst_against_ground_truth(
    burst_df: pd.DataFrame,
    ground_truth: List[dict],
) -> dict:
    """Evaluate burst detection against hand-labeled ground truth."""
    if not ground_truth:
        return {"error": "No burst ground truth provided"}

    detected = set()
    for _, row in burst_df[burst_df["is_burst"]].iterrows():
        day_str = (
            row["day"].strftime("%Y-%m-%d")
            if hasattr(row["day"], "strftime")
            else str(row["day"])[:10]
        )
        detected.add((day_str, row["country"]))

    tp, fp, fn, tn = 0, 0, 0, 0
    details = []

    for gt in ground_truth:
        key = (gt["day"], gt["country"])
        is_detected = key in detected
        is_real = gt.get("is_real_burst", True)

        if is_real and is_detected:
            tp += 1
            details.append({**gt, "result": "true_positive"})
        elif is_real and not is_detected:
            fn += 1
            details.append({**gt, "result": "false_negative"})
        elif not is_real and is_detected:
            fp += 1
            details.append({**gt, "result": "false_positive"})
        else:
            tn += 1
            details.append({**gt, "result": "true_negative"})

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_ground_truth": len(ground_truth),
        "details": details,
    }


def evaluate_burst_vs_baseline(
    burst_df: pd.DataFrame,
    ground_truth: List[dict],
) -> dict:
    """Compare our z-score burst detection vs top-5-percentile baseline."""
    our_result = evaluate_burst_against_ground_truth(burst_df, ground_truth)

    # Baseline: top 5% by event count
    threshold = burst_df["event_count"].quantile(0.95)
    baseline_detected = set()
    for _, row in burst_df[burst_df["event_count"] > threshold].iterrows():
        day_str = (
            row["day"].strftime("%Y-%m-%d")
            if hasattr(row["day"], "strftime")
            else str(row["day"])[:10]
        )
        baseline_detected.add((day_str, row["country"]))

    tp, fp, fn = 0, 0, 0
    for gt in ground_truth:
        key = (gt["day"], gt["country"])
        is_real = gt.get("is_real_burst", True)
        is_det = key in baseline_detected
        if is_real and is_det:
            tp += 1
        elif is_real and not is_det:
            fn += 1
        elif not is_real and is_det:
            fp += 1

    b_prec = tp / max(tp + fp, 1)
    b_rec = tp / max(tp + fn, 1)
    b_f1 = 2 * b_prec * b_rec / max(b_prec + b_rec, 1e-9)

    return {
        "our_method": our_result,
        "baseline_precision": round(b_prec, 3),
        "baseline_recall": round(b_rec, 3),
        "baseline_f1": round(b_f1, 3),
        "f1_improvement": round(our_result.get("f1", 0) - b_f1, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _chain_to_retrieved_ids(chain: dict) -> list:
    """Extract ordered list of retrieved event IDs from a chain result."""
    all_linked = chain.get("previous", []) + chain.get("next", [])
    return [int(e.get("GLOBALEVENTID", 0)) for e in all_linked]


def evaluate_chain_against_ground_truth(
    df: pd.DataFrame,
    chain_ground_truth: List[dict],
    top_n: int = 5,
    window_days: int = 7,
    model=None,
    scaler=None,
) -> dict:
    """Evaluate chain retrieval against hand-labeled ground truth."""
    if not chain_ground_truth:
        return {"error": "No chain ground truth provided"}

    results = []
    for gt in chain_ground_truth:
        anchor_id = gt["anchor_event_id"]
        relevant = set(gt.get("relevant_event_ids", []))

        if not relevant:
            continue

        chain = find_chain(
            df, anchor_id, window_days=window_days, top_n=top_n,
            model=model, scaler=scaler,
        )
        retrieved = _chain_to_retrieved_ids(chain)

        results.append({
            "anchor_event_id": anchor_id,
            "description": gt.get("description", ""),
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant),
            "precision_at_3": precision_at_k(retrieved, relevant, 3),
            "precision_at_5": precision_at_k(retrieved, relevant, 5),
            "recall_at_5": recall_at_k(retrieved, relevant, 5),
            "ndcg_at_5": ndcg_at_k(retrieved, relevant, 5),
            "mrr": mean_reciprocal_rank(retrieved, relevant),
            "map": mean_average_precision(retrieved, relevant),
            "pattern": chain.get("pattern", "Unknown"),
        })

    if not results:
        return {"error": "No valid ground truth entries matched"}

    return {
        "mean_precision_at_3": round(float(np.mean([r["precision_at_3"] for r in results])), 4),
        "mean_precision_at_5": round(float(np.mean([r["precision_at_5"] for r in results])), 4),
        "mean_recall_at_5": round(float(np.mean([r["recall_at_5"] for r in results])), 4),
        "mean_ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in results])), 4),
        "mean_mrr": round(float(np.mean([r["mrr"] for r in results])), 4),
        "mean_map": round(float(np.mean([r["map"] for r in results])), 4),
        "num_queries": len(results),
        "details": results,
    }


def evaluate_chain_sample(
    df: pd.DataFrame,
    sample_ids: list,
    window_days: int = 7,
    top_n: int = 5,
) -> list:
    """For a sample of events, build chains and compute quality metrics."""
    results = []
    for eid in sample_ids:
        chain = find_chain(df, int(eid), window_days, top_n)
        if chain["selected"] is None:
            continue

        all_linked = chain["previous"] + chain["next"]
        if not all_linked:
            results.append({
                "event_id": int(eid), "chain_size": 0,
                "actor_overlap": 0, "country_overlap": 0,
                "avg_chain_score": 0, "pattern": chain.get("pattern", "Unknown"),
            })
            continue

        anchor = chain["selected"]
        anchor_actors = {
            str(anchor.get("actor1_clean", "")),
            str(anchor.get("actor2_clean", "")),
        } - {"Unknown", ""}
        anchor_country = anchor.get("country", "")

        actor_hits = sum(
            1 for e in all_linked
            if str(e.get("actor1_clean", "")) in anchor_actors
            or str(e.get("actor2_clean", "")) in anchor_actors
        )
        country_hits = sum(
            1 for e in all_linked if e.get("country") == anchor_country
        )
        avg_score = sum(e.get("chain_score", 0) for e in all_linked) / len(all_linked)

        results.append({
            "event_id": int(eid),
            "chain_size": len(all_linked),
            "actor_overlap": round(actor_hits / len(all_linked), 3),
            "country_overlap": round(country_hits / len(all_linked), 3),
            "avg_chain_score": round(avg_score, 3),
            "pattern": chain.get("pattern", "Unknown"),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_pattern_detection(
    df: pd.DataFrame,
    chain_ground_truth: List[dict],
    top_n: int = 5,
    window_days: int = 7,
) -> dict:
    """Evaluate pattern classification accuracy against ground truth labels."""
    y_true = []
    y_pred = []

    for gt in chain_ground_truth:
        expected = gt.get("expected_pattern")
        if not expected:
            continue

        anchor_id = gt["anchor_event_id"]
        chain = find_chain(df, anchor_id, window_days=window_days, top_n=top_n)
        predicted = chain.get("pattern", "Unknown")

        y_true.append(expected)
        y_pred.append(predicted)

    if not y_true:
        return {"error": "No pattern labels in ground truth"}

    labels = sorted(set(y_true + y_pred))
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}

    # Confusion matrix
    cm = [[0] * n for _ in range(n)]
    correct = 0
    for t, p in zip(y_true, y_pred):
        cm[label_idx[t]][label_idx[p]] += 1
        if t == p:
            correct += 1

    accuracy = correct / len(y_true)

    # Macro F1
    f1_per_class = []
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1_per_class.append(f1)

    macro_f1 = sum(f1_per_class) / max(len(f1_per_class), 1)

    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "confusion_matrix": cm,
        "labels": labels,
        "n_evaluated": len(y_true),
        "per_class_f1": {l: round(f, 3) for l, f in zip(labels, f1_per_class)},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_GROUPS = {
    "full_model": [],
    "no_actor": ["actor_exact_match", "actor_country_match", "actor_country_interact"],
    "no_temporal": ["temporal_decay", "is_precursor", "is_followup"],
    "no_goldstein": ["goldstein_distance"],
    "no_location": ["same_location"],
    "no_importance": ["candidate_strength", "anchor_strength", "mention_ratio"],
    "no_cross_country": ["cross_country"],
}


def run_ablation_study(
    df: pd.DataFrame,
    chain_ground_truth: List[dict],
    window_days: int = 7,
    top_n: int = 5,
) -> dict:
    """
    Ablation: disable one feature group at a time, measure nDCG@5.
    Also tests random baseline and recency-only baseline.
    """
    results = {}

    # Standard ablations
    for name, removed_features in ABLATION_GROUPS.items():
        modified_w = HEURISTIC_WEIGHTS.copy()
        for feat in removed_features:
            idx = FEATURE_NAMES.index(feat)
            modified_w[idx] = 0.0

        ndcg_scores = _eval_with_weights(
            df, chain_ground_truth, modified_w, window_days, top_n,
        )
        results[name] = {
            "mean_ndcg_5": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0,
            "std_ndcg_5": round(float(np.std(ndcg_scores)), 4) if ndcg_scores else 0,
            "n_queries": len(ndcg_scores),
        }

    # Random baseline
    random_scores = _eval_random_baseline(df, chain_ground_truth, window_days, top_n)
    results["random_baseline"] = {
        "mean_ndcg_5": round(float(np.mean(random_scores)), 4) if random_scores else 0,
        "std_ndcg_5": round(float(np.std(random_scores)), 4) if random_scores else 0,
        "n_queries": len(random_scores),
    }

    # Recency-only baseline (only temporal_decay weight)
    recency_w = np.zeros_like(HEURISTIC_WEIGHTS)
    recency_w[FEATURE_NAMES.index("temporal_decay")] = 1.0
    recency_scores = _eval_with_weights(
        df, chain_ground_truth, recency_w, window_days, top_n,
    )
    results["recency_only"] = {
        "mean_ndcg_5": round(float(np.mean(recency_scores)), 4) if recency_scores else 0,
        "std_ndcg_5": round(float(np.std(recency_scores)), 4) if recency_scores else 0,
        "n_queries": len(recency_scores),
    }

    # Compute deltas
    full_ndcg = results["full_model"]["mean_ndcg_5"]
    for name, res in results.items():
        res["delta_vs_full"] = round(res["mean_ndcg_5"] - full_ndcg, 4)

    return results


def _eval_with_weights(
    df, chain_gt, weights, window_days, top_n,
) -> list[float]:
    """Evaluate chain retrieval with custom weight vector."""
    scores = []
    for gt in chain_gt:
        anchor_id = gt["anchor_event_id"]
        relevant = set(gt.get("relevant_event_ids", []))
        if not relevant:
            continue

        anchor = df[df["GLOBALEVENTID"] == anchor_id]
        if anchor.empty:
            continue
        anchor_row = anchor.iloc[0]
        anchor_date = anchor_row["event_date"]
        if pd.isna(anchor_date):
            continue

        start = anchor_date - timedelta(days=window_days)
        end = anchor_date + timedelta(days=window_days)
        mask = (
            (df["event_date"] >= start)
            & (df["event_date"] <= end)
            & (df["GLOBALEVENTID"] != anchor_id)
        )
        candidates = df[mask].copy()
        if candidates.empty:
            scores.append(0.0)
            continue

        F = compute_features_batch(anchor_row, candidates)
        candidates["_score"] = F @ weights
        top = candidates.nlargest(top_n * 2, "_score")
        retrieved = top["GLOBALEVENTID"].astype(int).tolist()
        scores.append(ndcg_at_k(retrieved, relevant, 5))

    return scores


def _eval_random_baseline(df, chain_gt, window_days, top_n, n_trials=5) -> list[float]:
    """Evaluate random selection as baseline."""
    all_scores = []
    for gt in chain_gt:
        anchor_id = gt["anchor_event_id"]
        relevant = set(gt.get("relevant_event_ids", []))
        if not relevant:
            continue

        anchor = df[df["GLOBALEVENTID"] == anchor_id]
        if anchor.empty:
            continue
        anchor_date = anchor.iloc[0]["event_date"]
        if pd.isna(anchor_date):
            continue

        start = anchor_date - timedelta(days=window_days)
        end = anchor_date + timedelta(days=window_days)
        pool = df[
            (df["event_date"] >= start)
            & (df["event_date"] <= end)
            & (df["GLOBALEVENTID"] != anchor_id)
        ]
        if pool.empty:
            all_scores.append(0.0)
            continue

        trial_scores = []
        for _ in range(n_trials):
            sample = pool.sample(min(top_n * 2, len(pool)))
            retrieved = sample["GLOBALEVENTID"].astype(int).tolist()
            trial_scores.append(ndcg_at_k(retrieved, relevant, 5))

        all_scores.append(float(np.mean(trial_scores)))

    return all_scores


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON (scored vs random, for display)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_vs_random_baseline(
    df: pd.DataFrame,
    sample_ids: list,
    window_days: int = 7,
    top_n: int = 5,
    n_random_trials: int = 5,
) -> dict:
    """Compare scored chains vs random selection — actor/country overlap lift."""
    scored_results = []
    random_results = []

    for eid in sample_ids:
        chain = find_chain(df, int(eid), window_days, top_n)
        if chain["selected"] is None:
            continue

        anchor = chain["selected"]
        anchor_country = anchor.get("country", "")
        anchor_actors = {
            str(anchor.get("actor1_clean", "")),
            str(anchor.get("actor2_clean", "")),
        } - {"Unknown", ""}

        all_linked = chain["previous"] + chain["next"]
        if all_linked:
            scored_actor_overlap = sum(
                1 for e in all_linked
                if str(e.get("actor1_clean", "")) in anchor_actors
                or str(e.get("actor2_clean", "")) in anchor_actors
            ) / len(all_linked)
            scored_country_overlap = sum(
                1 for e in all_linked if e.get("country") == anchor_country
            ) / len(all_linked)
            scored_results.append({
                "actor_overlap": scored_actor_overlap,
                "country_overlap": scored_country_overlap,
            })

        # Random baseline
        anchor_date = pd.to_datetime(str(anchor.get("event_date", ""))[:10])
        if pd.isna(anchor_date):
            continue
        start = anchor_date - timedelta(days=window_days)
        end = anchor_date + timedelta(days=window_days)
        pool = df[
            (df["event_date"] >= start) & (df["event_date"] <= end)
            & (df["GLOBALEVENTID"] != int(eid))
        ]
        if pool.empty:
            continue

        for _ in range(n_random_trials):
            n_sample = min(top_n * 2, len(pool))
            rand = pool.sample(n_sample).to_dict("records")
            r_actor = sum(
                1 for e in rand
                if str(e.get("actor1_clean", "")) in anchor_actors
                or str(e.get("actor2_clean", "")) in anchor_actors
            ) / len(rand)
            r_country = sum(
                1 for e in rand if e.get("country") == anchor_country
            ) / len(rand)
            random_results.append({"actor_overlap": r_actor, "country_overlap": r_country})

    if not scored_results or not random_results:
        return {"error": "Not enough data for baseline comparison"}

    return {
        "scored": {
            "mean_actor_overlap": round(np.mean([r["actor_overlap"] for r in scored_results]), 3),
            "mean_country_overlap": round(np.mean([r["country_overlap"] for r in scored_results]), 3),
        },
        "random_baseline": {
            "mean_actor_overlap": round(np.mean([r["actor_overlap"] for r in random_results]), 3),
            "mean_country_overlap": round(np.mean([r["country_overlap"] for r in random_results]), 3),
        },
        "improvement": {
            "actor_overlap_lift": round(
                np.mean([r["actor_overlap"] for r in scored_results])
                - np.mean([r["actor_overlap"] for r in random_results]), 3
            ),
            "country_overlap_lift": round(
                np.mean([r["country_overlap"] for r in scored_results])
                - np.mean([r["country_overlap"] for r in random_results]), 3
            ),
        },
        "n_scored": len(scored_results),
        "n_random_trials": len(random_results),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE TIME
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_response_time(df: pd.DataFrame, event_id: int, n_runs: int = 3) -> float:
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        find_chain(df, int(event_id))
        times.append(time.time() - t0)
    return round(sum(times) / len(times), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_evaluation_summary(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    cleaning_report: dict,
    ground_truth_path: str = None,
    model=None,
    scaler=None,
) -> dict:
    """Build complete evaluation summary."""
    summary = {
        "data_quality": evaluate_data_quality(df, cleaning_report),
        "burst_sanity": evaluate_burst_sanity(burst_df),
    }

    # Chain sample evaluation
    sample_ids = (
        df["GLOBALEVENTID"].dropna()
        .sample(min(15, len(df)), random_state=42)
        .tolist()
    )
    if sample_ids:
        summary["chain_evaluation"] = evaluate_chain_sample(df, sample_ids)
        summary["avg_response_time_sec"] = evaluate_response_time(df, sample_ids[0])
        summary["baseline_comparison"] = evaluate_vs_random_baseline(
            df, sample_ids[:5],
        )

    # Ground truth evaluation
    if ground_truth_path:
        gt_path = Path(ground_truth_path)
        if gt_path.exists():
            with open(gt_path) as f:
                gt_data = json.load(f)

            if "burst_ground_truth" in gt_data:
                summary["burst_ground_truth_eval"] = evaluate_burst_against_ground_truth(
                    burst_df, gt_data["burst_ground_truth"],
                )
                summary["burst_vs_baseline"] = evaluate_burst_vs_baseline(
                    burst_df, gt_data["burst_ground_truth"],
                )

            if "chain_ground_truth" in gt_data:
                summary["chain_ground_truth_eval"] = evaluate_chain_against_ground_truth(
                    df, gt_data["chain_ground_truth"], model=model, scaler=scaler,
                )
                summary["pattern_eval"] = evaluate_pattern_detection(
                    df, gt_data["chain_ground_truth"],
                )
                summary["ablation"] = run_ablation_study(
                    df, gt_data["chain_ground_truth"],
                )

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

def create_ground_truth_template(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    output_path: str,
    n_burst_samples: int = 15,
    n_chain_samples: int = 25,
) -> None:
    """Generate ground truth template pre-populated with labeling candidates."""
    template = {
        "instructions": (
            "Label each entry below. "
            "For bursts: set is_real_burst to true or false. "
            "For chains: list GLOBALEVENTIDs of events truly related to the anchor. "
            "Set expected_pattern to one of: Escalation, De-escalation, Persistence, Mixed."
        ),
        "burst_ground_truth": [],
        "chain_ground_truth": [],
    }

    # Burst candidates: top z-scores + some lower ones for negatives
    top_bursts = burst_df.nlargest(n_burst_samples, "z_score")
    for _, row in top_bursts.iterrows():
        template["burst_ground_truth"].append({
            "day": (
                row["day"].strftime("%Y-%m-%d")
                if hasattr(row["day"], "strftime")
                else str(row["day"])[:10]
            ),
            "country": row["country"],
            "event_count": int(row["event_count"]),
            "z_score": round(float(row["z_score"]), 2),
            "is_real_burst": True,
            "description": "TODO: describe what happened",
        })

    # Chain candidates: high-strength events, spread across countries
    high_strength = df.nlargest(n_chain_samples * 3, "event_strength")
    sample = high_strength.groupby("country").apply(
        lambda g: g.sample(min(n_chain_samples // 3 + 1, len(g)), random_state=42)
    ).reset_index(drop=True).head(n_chain_samples)

    for _, row in sample.iterrows():
        template["chain_ground_truth"].append({
            "anchor_event_id": int(row["GLOBALEVENTID"]),
            "date": (
                row["event_date"].strftime("%Y-%m-%d")
                if hasattr(row["event_date"], "strftime")
                else str(row["event_date"])[:10]
            ),
            "country": row["country"],
            "actors": f"{row.get('actor1_clean', '?')} → {row.get('actor2_clean', '?')}",
            "event_type": row.get("QuadLabel", "?"),
            "relevant_event_ids": [],
            "expected_pattern": "",
            "description": "TODO: describe the expected chain",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)
    print(f"  Ground truth template saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def save_evaluation(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  Evaluation saved → {path}")
