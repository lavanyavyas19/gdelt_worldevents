"""
data_cleaning.py
----------------
Cleans raw GDELT data:
  • type conversions
  • country filtering (US, IN, IR)
  • drops missing essential fields
  • creates helper date columns
"""

import pandas as pd
from typing import List

# Countries we care about
TARGET_COUNTRIES = {"US", "IN", "IR"}

# Columns we actually need (keeps memory small)
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

# Numeric columns that must be cast
NUMERIC_COLS = [
    "GLOBALEVENTID", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_Lat", "ActionGeo_Long",
]


def clean_data(
    df: pd.DataFrame,
    countries: set = TARGET_COUNTRIES,
    columns: List[str] = KEEP_COLUMNS,
) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Steps
    -----
    1. Keep only required columns
    2. Convert SQLDATE → datetime
    3. Cast numeric fields
    4. Filter to target countries
    5. Drop rows missing critical values
    6. Add year / month / day / week columns
    """

    # 1. Select columns (skip any that don't exist in the data)
    existing = [c for c in columns if c in df.columns]
    df = df[existing].copy()

    # 2. Date conversion
    df["SQLDATE"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["SQLDATE"])

    # 3. Numeric casting
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Country filter
    df = df[df["ActionGeo_CountryCode"].isin(countries)].copy()

    # 5. Drop rows missing QuadClass or coordinates
    df = df.dropna(subset=["QuadClass"])

    # 6. Helper date columns
    df["year"] = df["SQLDATE"].dt.year
    df["month"] = df["SQLDATE"].dt.to_period("M").astype(str)
    df["day"] = df["SQLDATE"].dt.date
    df["week"] = df["SQLDATE"].dt.to_period("W").astype(str)

    # Friendly country name
    country_map = {"US": "USA", "IN": "India", "IR": "Iran"}
    df["country"] = df["ActionGeo_CountryCode"].map(country_map)

    df = df.reset_index(drop=True)
    print(f"  Cleaned data: {len(df):,} rows")
    return df
