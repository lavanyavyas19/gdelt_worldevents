"""
prepare_data.py
---------------
End-to-end data pipeline: load → preprocess → burst detect → TF-IDF → evaluate → save.
Run once before launching the Streamlit dashboard.

Usage:
    python -m src.prepare_data
"""

import os
import sys
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    BURST_ROLLING_WINDOW, BURST_Z_THRESHOLD, BURST_MIN_EVENTS,
    ALLOWED_MONTHS, DATA_WINDOW_LABEL,
)
from src.data_loader import load_all_files
from src.preprocessing import preprocess
from src.burst import detect_bursts, save_burst_rules
from src.keywords import build_text_field, fit_tfidf
from src.storage import save_df
from src.evaluation import build_evaluation_summary, save_evaluation


def main():
    for d in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load raw files ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" GDELT Event Intelligence Pipeline")
    print(f" Analysis window: {DATA_WINDOW_LABEL}")
    print(f" Allowed months: {ALLOWED_MONTHS}")
    print(f"{'='*60}")

    print("\n[1/6] Loading raw GDELT files …")
    raw_df = load_all_files(RAW_DIR)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing (clean + engineer features) …")
    df, cleaning_report = preprocess(raw_df)

    # Save cleaning report
    report_path = os.path.join(OUTPUTS_DIR, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(cleaning_report, f, indent=2)
    print(f"  Cleaning report saved → {report_path}")

    # ── 3. Save processed events ──────────────────────────────────────────
    print("\n[3/6] Saving processed events …")
    path = save_df(df, os.path.join(PROCESSED_DIR, "events"))
    print(f"  Saved {os.path.basename(path)} ({len(df):,} rows)")

    # ── 4. Burst detection ────────────────────────────────────────────────
    print("\n[4/6] Detecting bursts …")
    burst_rules = {
        "rolling_window": BURST_ROLLING_WINDOW,
        "z_threshold": BURST_Z_THRESHOLD,
        "min_events": BURST_MIN_EVENTS,
    }
    save_burst_rules(burst_rules, os.path.join(MODELS_DIR, "burst_rules.json"))
    burst_df = detect_bursts(df, **burst_rules)
    save_df(burst_df, os.path.join(PROCESSED_DIR, "bursts"))

    # ── 5. TF-IDF ─────────────────────────────────────────────────────────
    print("\n[5/6] Fitting TF-IDF vectorizer …")
    texts = build_text_field(df)
    vectorizer = fit_tfidf(
        texts,
        save_path=os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    )
    n_features = len(vectorizer.get_feature_names_out())
    print(f"  Vocabulary size: {n_features}")

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    print("\n[6/6] Running evaluation …")
    eval_summary = build_evaluation_summary(df, burst_df, cleaning_report)
    save_evaluation(eval_summary, os.path.join(OUTPUTS_DIR, "evaluation_results.json"))

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Pipeline complete!")
    print(f" Events: {len(df):,} rows")
    print(f" Burst days: {int(burst_df['is_burst'].sum())}")
    print(f" TF-IDF features: {n_features}")
    print(f"{'='*60}")
    print(f"\n  Launch dashboard:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
