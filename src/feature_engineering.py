"""
feature_engineering.py
----------------------
Adds derived features to the cleaned GDELT dataframe:
  • QuadClass labels & simplified EventType
  • actor_pair
  • event_label (human-readable)
  • country_pair
"""

import pandas as pd

# ── QuadClass mappings ──────────────────────────────────────────────────────
QUAD_LABELS = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict",
}

QUAD_SIMPLE = {
    1: "Cooperation",
    2: "Cooperation",
    3: "Conflict",
    4: "Conflict",
}


def add_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map QuadClass to readable labels and a simplified EventType."""
    df["QuadLabel"] = df["QuadClass"].map(QUAD_LABELS)
    df["EventType"] = df["QuadClass"].map(QUAD_SIMPLE)
    return df


def add_actor_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Create actor_pair = 'Actor1 → Actor2'."""
    a1 = df["Actor1Name"].fillna("Unknown")
    a2 = df["Actor2Name"].fillna("Unknown")
    df["actor_pair"] = a1 + " → " + a2
    return df


def add_event_label_text(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable event label combining code + quad label."""
    df["event_label"] = (
        "Code "
        + df["EventRootCode"].astype(str)
        + " – "
        + df["QuadLabel"].fillna("Unknown")
    )
    return df


def add_country_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Pair of Actor1Country → Actor2Country."""
    c1 = df["Actor1CountryCode"].fillna("UNK")
    c2 = df["Actor2CountryCode"].fillna("UNK")
    df["country_pair"] = c1 + " → " + c2
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps."""
    df = add_event_labels(df)
    df = add_actor_pair(df)
    df = add_event_label_text(df)
    df = add_country_pair(df)
    print(f"  Features added. Columns: {len(df.columns)}")
    return df
