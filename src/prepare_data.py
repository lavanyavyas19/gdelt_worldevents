
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    BURST_ROLLING_WINDOW, BURST_Z_THRESHOLD, BURST_MIN_EVENTS,
    DATA_WINDOW_LABEL, INGEST_START_DATE, INGEST_END_DATE,
    TARGET_COUNTRY_CODES, TARGET_COUNTRY_NAMES,
)
from src.ingest import download_range
from src.data_loader import load_all_files
from src.preprocessing import preprocess
from src.burst import detect_bursts, save_burst_rules
from src.keywords import build_text_field, fit_tfidf
from src.storage import save_df
from src.evaluation import (
    build_evaluation_summary, save_evaluation, create_ground_truth_template,
)
from src.cross_country import compute_all_pairs


def main():
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" GDELT Event Intelligence Pipeline")
    print(f" {DATA_WINDOW_LABEL}")
    print(f"{'='*60}")


    csv_files = [f for f in os.listdir(RAW_DIR) if f.upper().endswith(".CSV")]
    if len(csv_files) < 90:
        print(f"\n[0/9] Downloading GDELT daily exports ({INGEST_START_DATE} → {INGEST_END_DATE})...")
        download_range(INGEST_START_DATE, INGEST_END_DATE, RAW_DIR)
    else:
        print(f"\n[0/9] Found {len(csv_files)} raw files — skipping download")

 
    print(f"\n[1/9] Loading raw GDELT files (filtered to {TARGET_COUNTRY_CODES})...")
    raw_df = load_all_files(RAW_DIR, country_codes=TARGET_COUNTRY_CODES)


    print("\n[2/9] Preprocessing (clean + feature engineer)...")
    df, cleaning_report = preprocess(raw_df)

    report_path = os.path.join(OUTPUTS_DIR, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(cleaning_report, f, indent=2)
    print(f"  Cleaning report saved → {report_path}")

  
    print("\n[3/9] Saving processed events...")
    path = save_df(df, os.path.join(PROCESSED_DIR, "events"))
    print(f"  Saved {os.path.basename(path)} ({len(df):,} rows)")

  
    print("\n[4/9] Detecting bursts...")
    burst_rules = {
        "rolling_window": BURST_ROLLING_WINDOW,
        "z_threshold": BURST_Z_THRESHOLD,
        "min_events": BURST_MIN_EVENTS,
    }
    save_burst_rules(burst_rules, os.path.join(MODELS_DIR, "burst_rules.json"))
    burst_df = detect_bursts(df, **burst_rules)
    save_df(burst_df, os.path.join(PROCESSED_DIR, "bursts"))
    print(f"  Burst days detected: {int(burst_df['is_burst'].sum())}")

 
    print("\n[5/9] Fitting TF-IDF vectorizer...")
    texts = build_text_field(df)
    vectorizer = fit_tfidf(
        texts, save_path=os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    )
    n_features = len(vectorizer.get_feature_names_out())
    print(f"  Vocabulary size: {n_features}")

 
    print("\n[6/9] Computing cross-country lead-lag correlations...")
    lag_results = compute_all_pairs(burst_df, TARGET_COUNTRY_NAMES)
    lag_path = os.path.join(OUTPUTS_DIR, "cross_country_results.json")
    with open(lag_path, "w") as f:
        json.dump(lag_results, f, indent=2, default=_json_convert)
    for r in lag_results:
        if "interpretation" in r:
            print(f"  {r['interpretation']}")

  
    gt_path = os.path.join(OUTPUTS_DIR, "ground_truth.json")
    model, scaler = None, None

    if os.path.exists(gt_path):
        print("\n[7/9] Training chain model from ground truth...")
        from src.chain_model import train_from_ground_truth, load_model
        result = train_from_ground_truth(
            df, gt_path,
            model_path=os.path.join(MODELS_DIR, "chain_model.pkl"),
            scaler_path=os.path.join(MODELS_DIR, "chain_scaler.pkl"),
        )
        if "error" not in result:
            model, scaler = load_model(
                os.path.join(MODELS_DIR, "chain_model.pkl"),
                os.path.join(MODELS_DIR, "chain_scaler.pkl"),
            )
            print(f"  Model trained successfully")
        else:
            print(f"  {result['error']}")
    else:
        print("\n[7/9] No ground truth — skipping model training (using heuristic scoring)")

 
    print("\n[8/9] Running evaluation...")
    eval_summary = build_evaluation_summary(
        df, burst_df, cleaning_report,
        ground_truth_path=gt_path if os.path.exists(gt_path) else None,
        model=model, scaler=scaler,
    )
    save_evaluation(eval_summary, os.path.join(OUTPUTS_DIR, "evaluation_results.json"))

  
    gt_template_path = os.path.join(OUTPUTS_DIR, "ground_truth_template.json")
    if not os.path.exists(gt_path):
        print("\n[9/9] Generating ground truth template for manual labeling...")
        create_ground_truth_template(df, burst_df, gt_template_path)
        print(f"  Fill in {gt_template_path}, rename to ground_truth.json, and re-run pipeline.")
    else:
        print("\n[9/9] Ground truth exists — evaluation includes retrieval/pattern/ablation metrics.")

   
    print(f"\n{'='*60}")
    print(f" Pipeline complete!")
    print(f" Events:     {len(df):,} rows")
    print(f" Date range: {cleaning_report.get('date_min', '?')} → {cleaning_report.get('date_max', '?')}")
    print(f" Countries:  {cleaning_report.get('countries', {})}")
    print(f" Burst days: {int(burst_df['is_burst'].sum())}")
    print(f" TF-IDF:     {n_features} features")
    print(f"{'='*60}")
    print(f"\n  Launch dashboard:  streamlit run app.py\n")


def _json_convert(obj):
    """JSON serializer for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()
