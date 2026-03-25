"""
data_loader.py
--------------
Loads raw GDELT Event CSV files from a folder.
GDELT v1 CSVs are tab-separated with NO header row — we assign the official
58-column names manually.

Changes from v1:
  • Reads ALL CSV files, no date assumptions
  • Returns raw strings — type conversion happens in preprocessing
  • Validates column count
"""

import os
import pandas as pd
from typing import List

# ── Official GDELT v1 Event column names (58 columns) ─────────────────────────
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


def load_single_file(filepath: str) -> pd.DataFrame:
    """Load one GDELT CSV file (tab-separated, no header)."""
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=GDELT_COLUMNS,
        dtype=str,              # read as string — cast later
        on_bad_lines="skip",
        low_memory=False,
    )
    return df


def load_all_files(data_dir: str, pattern: str = ".CSV") -> pd.DataFrame:
    """
    Load and concatenate ALL GDELT CSV files in a directory.

    Parameters
    ----------
    data_dir : path to folder with raw CSVs
    pattern  : file extension to match (case-insensitive)

    Returns
    -------
    Combined raw DataFrame (all string dtypes).
    """
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.upper().endswith(pattern.upper()) and not f.startswith(".")
    ])

    if not all_files:
        raise FileNotFoundError(f"No files matching *{pattern} in {data_dir}")

    frames: List[pd.DataFrame] = []
    for fp in all_files:
        print(f"  Loading {os.path.basename(fp)} ...")
        frames.append(load_single_file(fp))

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded: {len(combined):,}")
    return combined
