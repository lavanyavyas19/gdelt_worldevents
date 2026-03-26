"""
data_loader.py
--------------
Load raw GDELT Event CSV files with immediate country filtering.

GDELT daily export CSVs are tab-separated with NO header row.
We assign the official 58-column names and keep only the columns
and countries the system needs.
"""

import os
import pandas as pd
from typing import List, Set

# ── Official GDELT v1/v2-daily-export column names (58 columns) ─────────────
GDELT_COLUMNS: List[str] = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    # Actor 1
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    # Actor 2
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    # Event
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    # Actor 1 Geo
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    # Actor 2 Geo
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    # Action Geo
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    # Metadata
    "DATEADDED", "SOURCEURL",
]

# Columns the system actually uses — everything else is dropped on load.
KEEP_COLUMNS: List[str] = [
    "GLOBALEVENTID", "SQLDATE",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor1Type1Code",
    "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_Lat", "ActionGeo_Long",
    "SOURCEURL",
]


def load_single_file(
    filepath: str,
    country_codes: Set[str] | None = None,
) -> pd.DataFrame:
    """
    Load one GDELT CSV with immediate country filtering and column pruning.

    Parameters
    ----------
    filepath      : path to a tab-separated GDELT export CSV
    country_codes : if provided, keep only rows whose ActionGeo_CountryCode
                    is in this set (e.g. {"US", "IN", "IR"})
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=GDELT_COLUMNS,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )

    # Country filter first (biggest reduction)
    if country_codes:
        df = df[df["ActionGeo_CountryCode"].isin(country_codes)]

    # Column pruning
    existing = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[existing]

    return df


def load_all_files(
    data_dir: str,
    country_codes: Set[str] | None = None,
    batch_size: int = 15,
    pattern: str = ".CSV",
) -> pd.DataFrame:
    """
    Load and concatenate all GDELT CSVs in *data_dir*, filtering by country
    during load to keep memory usage low.

    Parameters
    ----------
    data_dir      : folder containing raw CSV exports
    country_codes : country filter (applied per-file)
    batch_size    : files per concat batch (controls peak memory)
    pattern       : file extension to match (case-insensitive)
    """
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.upper().endswith(pattern.upper()) and not f.startswith(".")
    ])

    if not all_files:
        raise FileNotFoundError(f"No files matching *{pattern} in {data_dir}")

    frames: List[pd.DataFrame] = []
    running_rows = 0

    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]
        batch_frames = []

        for fp in batch_files:
            df = load_single_file(fp, country_codes)
            batch_frames.append(df)
            running_rows += len(df)
            print(f"  {os.path.basename(fp):>30s}  {len(df):>7,} rows")

        frames.append(pd.concat(batch_frames, ignore_index=True))

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows loaded (filtered): {len(combined):,}")
    return combined
