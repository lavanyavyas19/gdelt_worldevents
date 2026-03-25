"""
preprocessing.py
-----------------
Unified cleaning + feature engineering pipeline.

This replaces both data_cleaning.py and feature_engineering.py with a single
robust module that:
  1. Selects and validates required columns
  2. Parses SQLDATE using exact %Y%m%d format → event_date
  3. Casts numeric columns
  4. Filters to target countries (US, IN, IR)
  5. Restricts to the Dec–Mar analysis window
  6. Cleans actor names
  7. Derives all needed features
  8. Removes duplicates
  9. Reports a cleaning summary
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import Dict, Any

from .config import (
    TARGET_COUNTRY_CODES, COUNTRY_CODE_MAP, ALLOWED_MONTHS,
    QUAD_LABELS, QUAD_SIMPLE, CAMEO_ROOT_LABELS,
)

# ── Columns we need from the raw data ─────────────────────────────────────────
KEEP_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_Lat", "ActionGeo_Long",
    "SOURCEURL",
]

NUMERIC_COLS = [
    "GLOBALEVENTID", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_Lat", "ActionGeo_Long",
]


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : raw DataFrame from data_loader (all string dtypes)

    Returns
    -------
    (cleaned_df, cleaning_report)
    """
    report: Dict[str, Any] = {"rows_raw": len(df)}

    # ── 1. Select columns ─────────────────────────────────────────────────
    existing = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing columns (will be skipped): {missing}")
    df = df[existing].copy()

    # ── 2. Parse SQLDATE → event_date ─────────────────────────────────────
    #    GDELT SQLDATE is YYYYMMDD as a string/int. We parse with exact format.
    df["event_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
    invalid_dates = df["event_date"].isna().sum()
    df = df.dropna(subset=["event_date"])
    report["invalid_dates_dropped"] = int(invalid_dates)

    # ── 3. Cast numeric columns ───────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Filter to target countries ─────────────────────────────────────
    before_country = len(df)
    df = df[df["ActionGeo_CountryCode"].isin(TARGET_COUNTRY_CODES)].copy()
    report["rows_outside_countries"] = before_country - len(df)

    # ── 5. Drop rows missing critical fields ──────────────────────────────
    df = df.dropna(subset=["QuadClass"])

    # ── 6. Restrict to Dec–Mar window ─────────────────────────────────────
    #    Strategy: identify the dominant year (most rows), then keep only
    #    Dec of (year-1) + Jan-Mar of (year) — and same for (year)/(year+1).
    #    This prevents stale rows from unrelated years (e.g. 2016) leaking
    #    through just because they happen to fall in an allowed month.
    df["_month_num"] = df["event_date"].dt.month
    df["_year"] = df["event_date"].dt.year
    before_window = len(df)

    dominant_year = int(df["_year"].mode().iloc[0])

    dec_mask = (df["_month_num"] == 12) & (df["_year"].isin([dominant_year - 1, dominant_year]))
    jan_mar_mask = (df["_month_num"].isin([1, 2, 3])) & (df["_year"].isin([dominant_year, dominant_year + 1]))
    df = df[dec_mask | jan_mar_mask].copy()

    report["rows_outside_window"] = before_window - len(df)
    report["dominant_year"] = dominant_year
    df = df.drop(columns=["_month_num", "_year"])

    # ── 7. Remove duplicates ──────────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["GLOBALEVENTID"])
    report["duplicates_removed"] = before_dedup - len(df)

    # ── 8. Clean actor names ──────────────────────────────────────────────
    df["actor1_clean"] = _clean_actor(df["Actor1Name"])
    df["actor2_clean"] = _clean_actor(df["Actor2Name"])
    df["has_missing_actor"] = (
        df["Actor1Name"].isna() | df["Actor2Name"].isna()
    )

    # ── 9. Derive date helper columns ─────────────────────────────────────
    df["year"] = df["event_date"].dt.year
    df["month"] = df["event_date"].dt.month
    df["month_label"] = df["event_date"].dt.to_period("M").astype(str)
    df["week_label"] = df["event_date"].dt.to_period("W").astype(str)
    df["day"] = df["event_date"].dt.normalize()  # datetime64 (midnight)

    # ── 10. Country name ──────────────────────────────────────────────────
    df["country"] = df["ActionGeo_CountryCode"].map(COUNTRY_CODE_MAP)

    # ── 11. Quad labels and event type ────────────────────────────────────
    df["QuadLabel"] = df["QuadClass"].map(QUAD_LABELS)
    df["EventType"] = df["QuadClass"].map(QUAD_SIMPLE)

    # CAMEO root code label
    df["EventRootLabel"] = df["EventRootCode"].astype(str).str.zfill(2).map(CAMEO_ROOT_LABELS)

    # ── 12. Actor pair and event label ────────────────────────────────────
    df["actor_pair"] = df["actor1_clean"] + " → " + df["actor2_clean"]
    df["event_label"] = (
        df["GLOBALEVENTID"].astype(int).astype(str)
        + " | "
        + df["event_date"].dt.strftime("%Y-%m-%d")
        + " | "
        + df["actor1_clean"]
        + " → "
        + df["actor2_clean"]
        + " | "
        + df["QuadLabel"].fillna("Unknown")
    )

    # ── 13. Event strength score ──────────────────────────────────────────
    #    Composite signal: combines mentions, sources, and tone magnitude
    df["event_strength"] = (
        np.log1p(df["NumMentions"].fillna(0))
        + np.log1p(df["NumSources"].fillna(0))
        + df["AvgTone"].abs() / 10
    ).round(3)

    # ── 14. Text field for TF-IDF ─────────────────────────────────────────
    df["text_for_tfidf"] = _build_tfidf_text(df)

    # ── 15. Source domain ─────────────────────────────────────────────────
    df["source_domain"] = df["SOURCEURL"].apply(_extract_domain)

    # ── 16. Country pair ──────────────────────────────────────────────────
    c1 = df["Actor1CountryCode"].fillna("UNK")
    c2 = df["Actor2CountryCode"].fillna("UNK")
    df["country_pair"] = c1 + " → " + c2

    # ── Final ─────────────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    report["rows_final"] = len(df)
    report["date_min"] = str(df["event_date"].min().date())
    report["date_max"] = str(df["event_date"].max().date())
    report["countries"] = df["country"].value_counts().to_dict()

    print(f"  Preprocessing complete: {len(df):,} rows")
    print(f"  Date range: {report['date_min']} → {report['date_max']}")
    print(f"  Countries: {report['countries']}")
    return df, report


# ── Helper functions ──────────────────────────────────────────────────────────

def _clean_actor(series: pd.Series) -> pd.Series:
    """Standardise actor names: title-case, strip, fill missing."""
    cleaned = (
        series
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )
    # Replace empty strings, literal "Nan", and "None"
    cleaned = cleaned.replace(
        {"": "Unknown", "Nan": "Unknown", "None": "Unknown", "Na": "Unknown"}
    )
    # Also catch any remaining case variations
    cleaned = cleaned.where(~cleaned.str.lower().isin(["nan", "none", ""]), "Unknown")
    return cleaned


def _build_tfidf_text(df: pd.DataFrame) -> pd.Series:
    """Combine multiple fields into a single text for keyword analysis."""
    parts = [
        df["actor1_clean"].fillna(""),
        df["actor2_clean"].fillna(""),
        df["ActionGeo_FullName"].fillna(""),
        df["QuadLabel"].fillna(""),
        df["EventRootLabel"].fillna(""),
    ]
    return parts[0].str.cat(parts[1:], sep=" ").str.strip()


def _extract_domain(url) -> str:
    """Extract domain from URL, return empty string on failure."""
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return ""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""
