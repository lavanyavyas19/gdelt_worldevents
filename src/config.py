"""
config.py
---------
Central configuration for the GDELT Event Intelligence Dashboard.
All magic numbers, paths, and constants live here.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ── Country configuration ─────────────────────────────────────────────────────
# ActionGeo_CountryCode → display name
COUNTRY_CODE_MAP = {"US": "USA", "IN": "India", "IR": "Iran"}
TARGET_COUNTRY_CODES = set(COUNTRY_CODE_MAP.keys())
TARGET_COUNTRY_NAMES = list(COUNTRY_CODE_MAP.values())

# ── Date window ───────────────────────────────────────────────────────────────
# We restrict the dataset to December through March only.
# These are the only months we keep, regardless of the year span in the data.
ALLOWED_MONTHS = [12, 1, 2, 3]
DATA_WINDOW_LABEL = "Dec – Mar analysis window"

# ── Burst detection defaults ──────────────────────────────────────────────────
BURST_ROLLING_WINDOW = 7
BURST_Z_THRESHOLD = 2.0
BURST_MIN_EVENTS = 5  # lowered — single-day exports have sparse data

# ── Event chain scoring weights ───────────────────────────────────────────────
CHAIN_SCORE_SAME_COUNTRY = 3
CHAIN_SCORE_SAME_ACTOR = 3
CHAIN_SCORE_SAME_EVENT_TYPE = 2
CHAIN_SCORE_SAME_QUAD = 2
CHAIN_SCORE_SAME_LOCATION = 1
CHAIN_SCORE_TONE_SIMILAR = 1
CHAIN_SCORE_TIME_PROXIMITY = 1  # scaled by closeness

# ── TF-IDF / keyword settings ────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 800
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)

# Geographic filler words to remove from TF-IDF (lowercased)
GEO_STOPWORDS = {
    "united", "states", "united states", "county", "district", "city",
    "province", "region", "state", "new", "north", "south", "east", "west",
    "central", "republic", "islamic", "democratic", "people", "general",
    "national", "international", "country", "world",
    # Country-specific filler (these get added dynamically per country filter)
    "india", "iran", "usa", "america", "american", "indian", "iranian",
    "columbia", "washington", "tehran", "delhi", "mumbai",
    # QuadLabel fragments and actor fill-ins that leak through
    "verbal", "material", "cooperation", "conflict", "unknown",
    "make", "public", "statement", "express", "intent",
}

# ── QuadClass mappings ────────────────────────────────────────────────────────
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

# ── CAMEO root code labels (top-level event families) ─────────────────────────
CAMEO_ROOT_LABELS = {
    "01": "Make Public Statement",
    "02": "Appeal",
    "03": "Express Intent to Cooperate",
    "04": "Consult",
    "05": "Diplomatic Cooperation",
    "06": "Material Cooperation",
    "07": "Provide Aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit Force Posture",
    "16": "Reduce Relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use Unconventional Mass Violence",
}

# ── UI constants ──────────────────────────────────────────────────────────────
COLOR_MAP_COUNTRY = {"USA": "#636EFA", "India": "#EF553B", "Iran": "#00CC96"}
COLOR_MAP_EVENT = {"Conflict": "#EF553B", "Cooperation": "#00CC96"}
COLOR_MAP_QUAD = {
    "Verbal Cooperation": "#00CC96",
    "Material Cooperation": "#19D3F3",
    "Verbal Conflict": "#FFA15A",
    "Material Conflict": "#EF553B",
}
MAX_MAP_POINTS = 15_000
EXPLORER_ROW_LIMIT = 1000
