
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.storage import load_df
from src.config import PROCESSED_DIR, OUTPUTS_DIR
from src.burst import detect_bursts
from src.chains import find_chain, HEURISTIC_WEIGHTS, FEATURE_NAMES
from src.chains import compute_features   



ANCHORS_PER_COUNTRY   = 8     
RELEVANCE_THRESHOLD   = 0.55   
WINDOW_DAYS           = 7
TOP_N                 = 10     


def _normalise_score(score: float) -> float:
    from src.config import CHAIN_MAX_POSSIBLE
    return min(score / max(CHAIN_MAX_POSSIBLE, 1), 1.0)


def generate(output_path: str | None = None) -> dict:
    """Generate and save ground truth. Returns the dict."""
    print("Loading events and burst data...")
    df = load_df(os.path.join(PROCESSED_DIR, "events"))
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["day"]        = pd.to_datetime(df["day"])

    burst_df = detect_bursts(df, rolling_window=7, z_threshold=2.0)
    burst_days = burst_df[burst_df["is_burst"]].copy()

    countries = df["country"].dropna().unique().tolist()

    burst_gt   = []
    chain_gt   = []

    for country in countries:
        print(f"  Processing {country}...")
        c_bursts = (
            burst_days[burst_days["country"] == country]
            .sort_values("z_score", ascending=False)
            .head(ANCHORS_PER_COUNTRY)
        )

        for _, b_row in c_bursts.iterrows():
            day_str = b_row["day"].strftime("%Y-%m-%d")
            z_score = float(b_row["z_score"])
            count   = int(b_row["event_count"])

            
            is_real = bool(z_score >= 2.0 and count >= 10)
            burst_gt.append({
                "day"          : day_str,
                "country"      : country,
                "event_count"  : count,
                "z_score"      : round(z_score, 3),
                "is_real_burst": is_real,
                "description"  : f"Auto-labeled: z={z_score:.2f}, count={count}",
            })

            
            day_ts   = b_row["day"]
            day_mask = (df["day"] == day_ts) & (df["country"] == country)
            day_df   = df[day_mask]
            if day_df.empty:
                continue

            sort_col = "NumMentions" if "NumMentions" in day_df.columns else "GLOBALEVENTID"
            anchor_row = day_df.sort_values(sort_col, ascending=False).iloc[0]
            anchor_id  = int(anchor_row["GLOBALEVENTID"])
            anchor_date = anchor_row["event_date"]

            
            result = find_chain(
                df, anchor_id,
                window_days=WINDOW_DAYS,
                top_n=TOP_N,
                country_filter=country,
            )

            
            relevant_ids = []
            for section in ("previous", "next"):
                for evt in result.get(section, []):
                    raw_score = evt.get("chain_score", 0)
                    if _normalise_score(raw_score) >= RELEVANCE_THRESHOLD:
                        eid = evt.get("GLOBALEVENTID")
                        if eid is not None:
                            relevant_ids.append(int(eid))

            
            pattern = result.get("pattern", "")
            if not pattern or pattern == "Stable":
                pattern = "Persistence"    

            actors = (
                f"{anchor_row.get('actor1_clean','?')} → "
                f"{anchor_row.get('actor2_clean','?')}"
            )
            event_type = anchor_row.get("QuadLabel", anchor_row.get("EventType", "Unknown"))

            chain_gt.append({
                "anchor_event_id"  : anchor_id,
                "date"             : day_str,
                "country"          : country,
                "actors"           : actors,
                "event_type"       : str(event_type),
                "relevant_event_ids": relevant_ids,
                "expected_pattern" : str(pattern),
                "description"      : (
                    f"Auto-labeled via heuristic chain scorer. "
                    f"z={z_score:.2f}, {len(relevant_ids)} relevant events found."
                ),
            })

    ground_truth = {
        "instructions": (
            "Auto-generated ground truth using heuristic chain scoring. "
            "Labels are 'silver standard'. Manual review can improve quality. "
            "burst_ground_truth: is_real_burst flags. "
            "chain_ground_truth: relevant_event_ids are events with heuristic score >= 0.55."
        ),
        "generated_at"    : pd.Timestamp.now().isoformat(),
        "n_burst_labels"  : len(burst_gt),
        "n_chain_labels"  : len(chain_gt),
        "burst_ground_truth": burst_gt,
        "chain_ground_truth": chain_gt,
    }

    out_path = output_path or os.path.join(OUTPUTS_DIR, "ground_truth.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"  Saved → {out_path}")
    print(f"  Burst labels: {len(burst_gt)}   Chain labels: {len(chain_gt)}")
    return ground_truth


if __name__ == "__main__":
    generate()
