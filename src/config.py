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

# ── Data ingestion ───────────────────────────────────────────────────────────
INGEST_START_DATE = "2025-12-01"
INGEST_END_DATE = "2026-03-31"

# ── Country configuration ─────────────────────────────────────────────────────
# ActionGeo_CountryCode → display name
COUNTRY_CODE_MAP = {"US": "USA", "IN": "India", "IR": "Iran"}
TARGET_COUNTRY_CODES = set(COUNTRY_CODE_MAP.keys())
TARGET_COUNTRY_NAMES = list(COUNTRY_CODE_MAP.values())

# ── Date window ───────────────────────────────────────────────────────────────
ALLOWED_MONTHS = [12, 1, 2, 3]
DATA_WINDOW_LABEL = "Dec 2025 – Mar 2026 analysis window"
# Hard ceiling: no event dated beyond this is shown anywhere in the app
DATA_CUTOFF_DATE = "2026-03-26"

# ── Burst detection defaults ──────────────────────────────────────────────────
BURST_ROLLING_WINDOW = 7
BURST_Z_THRESHOLD = 2.0
BURST_MIN_EVENTS = 5

# ── Event chain scoring ──────────────────────────────────────────────────────
# Legacy constants (kept for backward compat with any UI code that references them)
CHAIN_SCORE_SAME_COUNTRY = 3
CHAIN_SCORE_SAME_ACTOR = 3
CHAIN_SCORE_SAME_EVENT_TYPE = 2
CHAIN_SCORE_SAME_QUAD = 2
CHAIN_SCORE_SAME_LOCATION = 1
CHAIN_SCORE_TONE_SIMILAR = 1
CHAIN_SCORE_TIME_PROXIMITY = 1
CHAIN_SCORE_GOLDSTEIN_SIMILAR = 1
CHAIN_SCORE_EVENT_IMPORTANCE = 1
CHAIN_MAX_POSSIBLE = 18

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
    "india", "iran", "usa", "america", "american", "indian", "iranian",
    "columbia", "washington", "tehran", "delhi", "mumbai",
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

# ── Intelligence Hub (RAG / Summarizer / Q&A) ─────────────────────────────────
RAG_DIR             = os.path.join(PROJECT_ROOT, "data", "rag")
ARTICLE_CACHE_DIR   = os.path.join(PROJECT_ROOT, "data", "article_cache")
RAG_MAX_ARTICLES    = 10      # articles fetched per burst
RAG_CHUNK_CHARS     = 500     # characters per text chunk
RAG_CHUNK_OVERLAP   = 80      # overlap between chunks
RAG_TOP_K           = 4       # chunks returned per query
EMBED_MODEL         = "all-MiniLM-L6-v2"   # sentence-transformers model (384-dim)
OLLAMA_URL          = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL= "mistral"             # preferred local LLM
