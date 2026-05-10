"""
keywords.py
-----------
Improved TF-IDF keyword extraction with:
  • Custom geographic + filler stopword removal
  • Dynamic country-specific stopwords
  • Unigram + bigram support (configurable)
  • Burst vs normal keyword comparison
  • Meaningful phrase preservation

Replaces the old tfidf_module.py.
"""

import re
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .config import (
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE, GEO_STOPWORDS,
)

# ── Extended stopword list ────────────────────────────────────────────────────
# English stopwords + low-information tokens commonly found in GDELT
STOP_WORDS = set(
    "a an the is was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under and "
    "but or nor not so yet both either neither each every all any few more "
    "most other some such no only own same than too very that this these those "
    "i me my we our you your he him his she her it its they them their what "
    "which who whom how when where why there here are am about also just if "
    "then because while although since until up down said say says make made "
    "known said like would could".split()
)

# Merge with geographic stopwords
ALL_STOPWORDS = STOP_WORDS | GEO_STOPWORDS

# Country-specific words to remove when that country is already selected
COUNTRY_EXTRA_STOPS = {
    "USA": {"usa", "us", "america", "american", "americans", "united states",
            "washington", "new york", "california", "texas", "florida"},
    "India": {"india", "indian", "indians", "delhi", "new delhi", "mumbai",
              "bangalore", "chennai", "kolkata", "hyderabad"},
    "Iran": {"iran", "iranian", "iranians", "tehran", "isfahan", "shiraz",
             "islamic republic"},
}


def clean_text_for_tfidf(text: str, extra_stops: set = None) -> str:
    """
    Clean a single text string for TF-IDF analysis.
    - Lowercase
    - Remove punctuation
    - Remove stopwords + geographic filler
    - Optionally remove country-specific words
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stops = ALL_STOPWORDS
    if extra_stops:
        stops = stops | extra_stops

    tokens = [w for w in text.split() if w not in stops and len(w) > 1]
    return " ".join(tokens)


def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine relevant fields into a single text for keyword analysis."""
    parts = [
        df["actor1_clean"].fillna("") if "actor1_clean" in df.columns
        else df["Actor1Name"].fillna(""),
        df["actor2_clean"].fillna("") if "actor2_clean" in df.columns
        else df["Actor2Name"].fillna(""),
        df["ActionGeo_FullName"].fillna(""),
        df["QuadLabel"].fillna(""),
        df["EventRootLabel"].fillna("") if "EventRootLabel" in df.columns
        else df["EventRootCode"].astype(str),
    ]
    return parts[0].str.cat(parts[1:], sep=" ").str.strip()


def fit_tfidf(
    texts: pd.Series,
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int = TFIDF_MIN_DF,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    save_path: str = None,
):
    """
    Fit a TF-IDF vectorizer on pre-cleaned texts.
    Uses sklearn if available.
    """
    # Clean texts globally (no country filter)
    cleaned = texts.apply(lambda t: clean_text_for_tfidf(t))

    if HAS_SKLEARN:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,  # we already removed stopwords
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,  # use log(1 + tf) for better results
        )
    else:
        # Fallback — use simple implementation
        vectorizer = _SimpleTfidfVectorizer(
            max_features=max_features, min_df=min_df
        )

    vectorizer.fit(cleaned)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"  TF-IDF vectorizer saved → {save_path}")

    return vectorizer


def top_keywords(
    texts: pd.Series,
    vectorizer,
    top_n: int = 15,
    extra_stops: set = None,
) -> list:
    """
    Extract top-N keywords from a set of texts using the fitted vectorizer.

    Parameters
    ----------
    texts       : raw text series (will be cleaned internally)
    vectorizer  : fitted TfidfVectorizer
    top_n       : how many keywords to return
    extra_stops : additional words to remove (e.g., country name when filtering)
    """
    if texts.empty:
        return []

    cleaned = texts.apply(lambda t: clean_text_for_tfidf(t, extra_stops))
    matrix = vectorizer.transform(cleaned)
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(matrix, "toarray"):
        mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
    else:
        mean_scores = matrix.mean(axis=0)

    # Get top indices
    top_idx = mean_scores.argsort()[::-1][:top_n]
    return [
        {"keyword": feature_names[i], "score": round(float(mean_scores[i]), 5)}
        for i in top_idx
        if mean_scores[i] > 0
    ]


def keywords_by_country(
    df: pd.DataFrame, vectorizer, top_n: int = 15
) -> dict:
    """Extract top keywords per country, with country-specific stopword removal."""
    result = {}
    for country, grp in df.groupby("country"):
        texts = build_text_field(grp)
        extra_stops = COUNTRY_EXTRA_STOPS.get(country, set())
        result[country] = top_keywords(texts, vectorizer, top_n, extra_stops)
    return result


def keywords_for_bursts(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    vectorizer,
    top_n: int = 15,
) -> dict:
    """Extract keywords specifically during burst periods, per country."""
    burst_only = burst_df[burst_df["is_burst"]][["day", "country"]].copy()
    if burst_only.empty:
        return {}

    merged = df.merge(burst_only, on=["day", "country"], how="inner")
    result = {}
    for country, grp in merged.groupby("country"):
        texts = build_text_field(grp)
        extra_stops = COUNTRY_EXTRA_STOPS.get(country, set())
        result[country] = top_keywords(texts, vectorizer, top_n, extra_stops)
    return result


def keywords_normal_vs_burst(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    vectorizer,
    top_n: int = 15,
) -> dict:
    """
    Compare keywords during normal periods vs burst periods, per country.
    Returns dict[country] = {"normal": [...], "burst": [...]}
    """
    burst_days = set(
        zip(
            burst_df[burst_df["is_burst"]]["day"],
            burst_df[burst_df["is_burst"]]["country"],
        )
    )
    if not burst_days:
        return {}

    # Merge-based approach: O(n log n) instead of row-wise O(n²)
    burst_marker = burst_df[burst_df["is_burst"]][["day", "country"]].copy()
    burst_marker["_is_burst"] = True
    df_marked = df.merge(burst_marker, on=["day", "country"], how="left")
    df_marked["_is_burst"] = df_marked["_is_burst"].fillna(False)

    result = {}
    for country, grp in df_marked.groupby("country"):
        extra_stops = COUNTRY_EXTRA_STOPS.get(country, set())

        # Split into burst vs normal using pre-computed merge column
        is_burst_mask = grp["_is_burst"].astype(bool)
        burst_texts = build_text_field(grp.loc[is_burst_mask])
        normal_texts = build_text_field(grp.loc[~is_burst_mask])
        
        result[country] = {
            "burst": top_keywords(burst_texts, vectorizer, top_n, extra_stops),
            "normal": top_keywords(normal_texts, vectorizer, top_n, extra_stops),
        }
    return result


# ── Minimal fallback TF-IDF (when sklearn not installed) ──────────────────────

import math

class _SimpleTfidfVectorizer:
    """Lightweight TF-IDF for environments without sklearn."""

    def __init__(self, max_features=500, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_ = {}
        self.idf_ = None

    def fit(self, texts):
        docs = [t.split() for t in texts]
        n_docs = len(docs)
        df_counter = Counter()
        for doc in docs:
            df_counter.update(set(doc))
        candidates = {w: df for w, df in df_counter.items() if df >= self.min_df}
        top_words = sorted(candidates, key=candidates.get, reverse=True)[:self.max_features]
        self.vocabulary_ = {w: i for i, w in enumerate(top_words)}
        self.idf_ = np.zeros(len(self.vocabulary_))
        for w, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + candidates[w])) + 1
        return self

    def transform(self, texts):
        n = len(self.vocabulary_)
        rows = []
        for text in texts:
            tokens = text.split()
            tf = Counter(tokens)
            vec = np.zeros(n)
            for w, idx in self.vocabulary_.items():
                if w in tf:
                    vec[idx] = tf[w]
            total = sum(tf.values()) or 1
            vec = (vec / total) * self.idf_
            rows.append(vec)
        return np.array(rows)

    def get_feature_names_out(self):
        names = [""] * len(self.vocabulary_)
        for w, i in self.vocabulary_.items():
            names[i] = w
        return np.array(names)
