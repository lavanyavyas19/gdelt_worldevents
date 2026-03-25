"""
prepare_data.py
---------------
Orchestrates the full pipeline: load → clean → engineer → aggregate → save.
Run this script once before launching the Streamlit app.

Usage:
    python -m src.prepare_data
"""

import os
import sys
import pickle

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_all_files
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.aggregation import aggregate_by
from src.detect_bursts import detect_bursts, save_burst_rules, DEFAULT_RULES
from src.tfidf_module import build_text_field, fit_tfidf
from src.storage import save_df
from src.evaluation import (
    evaluate_chain_sample,
    evaluate_burst_sanity,
    evaluate_response_time,
    save_evaluation,
)


def main():
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # ── 1. Load ─────────────────────────────────────────────────────────
    print("\n[1/7] Loading raw GDELT files …")
    raw_df = load_all_files(raw_dir)

    # ── 2. Clean ────────────────────────────────────────────────────────
    print("\n[2/7] Cleaning data …")
    clean_df = clean_data(raw_df)

    # ── 3. Feature engineering ──────────────────────────────────────────
    print("\n[3/7] Engineering features …")
    df = engineer_features(clean_df)

    # ── 4. Save processed data ──────────────────────────────────────────
    print("\n[4/7] Saving processed data …")
    path = save_df(df, os.path.join(processed_dir, "events"))
    print(f"  Saved {os.path.basename(path)} ({len(df):,} rows)")

    # ── 5. Aggregation ──────────────────────────────────────────────────
    print("\n[5/7] Aggregating …")
    monthly = aggregate_by(df, "month")
    weekly = aggregate_by(df, "week")
    save_df(monthly, os.path.join(processed_dir, "agg_monthly"))
    save_df(weekly, os.path.join(processed_dir, "agg_weekly"))
    print(f"  Monthly: {len(monthly)} rows | Weekly: {len(weekly)} rows")

    # ── 6. Burst detection ──────────────────────────────────────────────
    print("\n[6/7] Detecting bursts …")
    save_burst_rules(DEFAULT_RULES, os.path.join(models_dir, "burst_rules.json"))
    burst_df = detect_bursts(df, DEFAULT_RULES)
    save_df(burst_df, os.path.join(processed_dir, "bursts"))

    # ── 7. TF-IDF ──────────────────────────────────────────────────────
    print("\n[7/7] Fitting TF-IDF …")
    texts = build_text_field(df)
    vectorizer = fit_tfidf(
        texts,
        max_features=500,
        save_path=os.path.join(models_dir, "tfidf_vectorizer.pkl"),
    )

    # ── Evaluation (optional but recommended) ───────────────────────────
    print("\n[Eval] Running evaluation …")
    sample_ids = df["GLOBALEVENTID"].dropna().sample(min(20, len(df))).tolist()
    chain_eval = evaluate_chain_sample(df, sample_ids)
    burst_eval = evaluate_burst_sanity(burst_df)
    if sample_ids:
        resp_time = evaluate_response_time(df, sample_ids[0])
    else:
        resp_time = 0

    save_evaluation(
        {
            "chain_evaluation": chain_eval,
            "burst_sanity": burst_eval,
            "avg_response_time_sec": resp_time,
        },
        os.path.join(outputs_dir, "evaluation_results.json"),
    )

    print("\n✓ Pipeline complete! You can now run: streamlit run app.py\n")


if __name__ == "__main__":
    main()
