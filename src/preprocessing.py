"""
preprocessing.py
-----------------
Unified cleaning + feature engineering pipeline.

Steps:
  1. Parse SQLDATE → event_date
  2. Cast numeric columns
  3. Filter to target countries + date window
  4. Remove duplicates
  5. Clean actor names
  6. Derive all features (country, quad labels, event_strength, etc.)
  7. Save cleaning report
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import Dict, Any

from .config import (
    TARGET_COUNTRY_CODES, COUNTRY_CODE_MAP, ALLOWED_MONTHS,
    QUAD_LABELS, QUAD_SIMPLE, CAMEO_ROOT_LABELS, DATA_CUTOFF_DATE,
)

_CUTOFF = pd.Timestamp(DATA_CUTOFF_DATE)

NUMERIC_COLS = [
    "GLOBALEVENTID", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_Lat", "ActionGeo_Long",
]


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full preprocessing pipeline.

    Returns (cleaned_df, cleaning_report).
    """
    report: Dict[str, Any] = {"rows_raw": len(df)}

    # ── 1. Parse SQLDATE → event_date ──────────────────────────────────────
    df["event_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
    invalid_dates = int(df["event_date"].isna().sum())
    df = df.dropna(subset=["event_date"])
    report["invalid_dates_dropped"] = invalid_dates

    # Hard ceiling — no future data beyond the analysis cutoff
    before_cutoff = len(df)
    df = df[df["event_date"] <= _CUTOFF].copy()
    report["rows_beyond_cutoff"] = before_cutoff - len(df)

    # ── 2. Cast numeric columns ────────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3. Filter to target countries ──────────────────────────────────────
    before_country = len(df)
    df = df[df["ActionGeo_CountryCode"].isin(TARGET_COUNTRY_CODES)].copy()
    report["rows_outside_countries"] = before_country - len(df)

    # ── 4. Drop rows missing critical fields ───────────────────────────────
    df = df.dropna(subset=["QuadClass"])

    # ── 5. Restrict to Dec–Mar window ──────────────────────────────────────
    month_num = df["event_date"].dt.month
    year_col = df["event_date"].dt.year
    before_window = len(df)

    dominant_year = int(year_col.mode().iloc[0])

    dec_mask = (month_num == 12) & (year_col.isin([dominant_year - 1, dominant_year]))
    jan_mar_mask = (month_num.isin([1, 2, 3])) & (year_col.isin([dominant_year, dominant_year + 1]))
    df = df[dec_mask | jan_mar_mask].copy()

    report["rows_outside_window"] = before_window - len(df)
    report["dominant_year"] = dominant_year

    # ── 6. Remove duplicates ───────────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["GLOBALEVENTID"])
    report["duplicates_removed"] = before_dedup - len(df)

    # ── 7. Clean actor names ───────────────────────────────────────────────
    df["actor1_clean"] = _clean_actor(df["Actor1Name"])
    df["actor2_clean"] = _clean_actor(df["Actor2Name"])
    df["has_missing_actor"] = df["Actor1Name"].isna() | df["Actor2Name"].isna()

    # ── 8. Date helper columns ─────────────────────────────────────────────
    df["year"] = df["event_date"].dt.year
    df["month"] = df["event_date"].dt.month
    df["month_label"] = df["event_date"].dt.to_period("M").astype(str)
    df["week_label"] = df["event_date"].dt.to_period("W").astype(str)
    df["day"] = df["event_date"].dt.normalize()

    # ── 9. Country name ────────────────────────────────────────────────────
    df["country"] = df["ActionGeo_CountryCode"].map(COUNTRY_CODE_MAP)

    # ── 10. Quad labels and event type ─────────────────────────────────────
    df["QuadLabel"] = df["QuadClass"].map(QUAD_LABELS)
    df["EventType"] = df["QuadClass"].map(QUAD_SIMPLE)
    df["EventRootLabel"] = (
        df["EventRootCode"].astype(str).str.zfill(2).map(CAMEO_ROOT_LABELS)
    )

    # ── 11. Actor pair and event label ─────────────────────────────────────
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

    # ── 12. Event strength (composite signal) ──────────────────────────────
    df["event_strength"] = (
        np.log1p(df["NumMentions"].fillna(0))
        + np.log1p(df["NumSources"].fillna(0))
        + df["AvgTone"].abs() / 10
        + df["GoldsteinScale"].abs().fillna(0) / 10
    ).round(3)

    # ── 13. Text field for TF-IDF ──────────────────────────────────────────
    df["text_for_tfidf"] = _build_tfidf_text(df)

    # ── 14. Source domain ──────────────────────────────────────────────────
    df["source_domain"] = df["SOURCEURL"].apply(_extract_domain)

    # ── 15. Country pair ───────────────────────────────────────────────────
    c1 = df["Actor1CountryCode"].fillna("UNK")
    c2 = df["Actor2CountryCode"].fillna("UNK")
    df["country_pair"] = c1 + " → " + c2

    # ── Final ──────────────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    report["rows_final"] = len(df)
    report["date_min"] = str(df["event_date"].min().date())
    report["date_max"] = str(df["event_date"].max().date())
    report["countries"] = df["country"].value_counts().to_dict()

    print(f"  Preprocessing complete: {len(df):,} rows")
    print(f"  Date range: {report['date_min']} → {report['date_max']}")
    print(f"  Countries: {report['countries']}")
    return df, report


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_actor(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("Unknown").astype(str).str.strip().str.title()
    )
    cleaned = cleaned.replace(
        {"": "Unknown", "Nan": "Unknown", "None": "Unknown", "Na": "Unknown"}
    )
    cleaned = cleaned.where(
        ~cleaned.str.lower().isin(["nan", "none", ""]), "Unknown"
    )
    return cleaned


def _build_tfidf_text(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["actor1_clean"].fillna(""),
        df["actor2_clean"].fillna(""),
        df["ActionGeo_FullName"].fillna(""),
        df["QuadLabel"].fillna(""),
        df["EventRootLabel"].fillna(""),
    ]
    return parts[0].str.cat(parts[1:], sep=" ").str.strip()


def _extract_domain(url) -> str:
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return ""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""
