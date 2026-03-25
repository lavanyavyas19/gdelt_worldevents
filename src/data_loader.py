"""
data_loader.py
--------------
Loads raw GDELT Event CSV files from a folder.
GDELT CSVs have NO headers — we assign the official 58-column names manually.
"""

import os
import pandas as pd
from typing import List, Optional

# ── Official GDELT v1 Event column names (58 columns) ──────────────────────
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
        dtype=str,            # read everything as string first
        on_bad_lines="skip",  # skip malformed rows
        low_memory=False,
    )
    return df


def load_all_files(data_dir: str, pattern: str = ".CSV") -> pd.DataFrame:
    """
    Load all GDELT CSV files from a directory and merge them.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing raw CSV files.
    pattern : str
        File extension to match (default: '.CSV').

    Returns
    -------
    pd.DataFrame
        Combined dataframe of all loaded files.
    """
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.upper().endswith(pattern.upper())
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
