"""
tfidf_module.py
---------------
Build a text field from event metadata, then extract top keywords
per country and during burst periods using TF-IDF.

Uses sklearn if available; otherwise falls back to a lightweight
pure-Python/numpy implementation.
"""

import pickle
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── Try sklearn first, fall back to built-in ────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _SklearnTfidf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Lightweight fallback TF-IDF ────────────────────────────────────────────
STOP_WORDS = set(
    "a an the is was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under and "
    "but or nor not so yet both either neither each every all any few more "
    "most other some such no only own same than too very that this these those "
    "i me my we our you your he him his she her it its they them their what "
    "which who whom how when where why there here are am about also just if "
    "then because while although since until up down".split()
)


def _tokenize(text: str) -> list:
    return [w for w in re.findall(r"[a-zA-Z]{2,}", text.lower()) if w not in STOP_WORDS]


class SimpleTfidfVectorizer:
    """Minimal TF-IDF vectorizer using only numpy + builtins."""

    def __init__(self, max_features=500, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_ = {}
        self.idf_ = None

    def fit(self, texts):
        docs = [_tokenize(t) for t in texts]
        n_docs = len(docs)

        # Document frequency
        df_counter = Counter()
        for doc in docs:
            df_counter.update(set(doc))

        # Filter by min_df and take top features by df
        candidates = {
            w: df for w, df in df_counter.items() if df >= self.min_df
        }
        top_words = sorted(candidates, key=candidates.get, reverse=True)[: self.max_features]
        self.vocabulary_ = {w: i for i, w in enumerate(top_words)}

        # IDF
        self.idf_ = np.zeros(len(self.vocabulary_))
        for w, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + candidates[w])) + 1
        return self

    def transform(self, texts):
        n = len(self.vocabulary_)
        rows = []
        for text in texts:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            vec = np.zeros(n)
            for w, idx in self.vocabulary_.items():
                if w in tf:
                    vec[idx] = tf[w]
            # Normalize TF
            total = sum(tf.values()) or 1
            vec = (vec / total) * self.idf_
            rows.append(vec)
        return np.array(rows)

    def get_feature_names_out(self):
        names = [""] * len(self.vocabulary_)
        for w, i in self.vocabulary_.items():
            names[i] = w
        return np.array(names)


# ── Public API ──────────────────────────────────────────────────────────────

def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine Actor1Name, Actor2Name, location, and event label into text."""
    parts = [
        df["Actor1Name"].fillna(""),
        df["Actor2Name"].fillna(""),
        df["ActionGeo_FullName"].fillna(""),
        df["QuadLabel"].fillna(""),
        df["EventRootCode"].astype(str),
    ]
    return parts[0].str.cat(parts[1:], sep=" ").str.strip()


def fit_tfidf(texts: pd.Series, max_features: int = 500, save_path: str = None, min_df: int = 2):
    """Fit a TF-IDF vectorizer (sklearn or fallback)."""
    if HAS_SKLEARN:
        vectorizer = _SklearnTfidf(
            max_features=max_features, stop_words="english",
            ngram_range=(1, 2), min_df=min_df,
        )
    else:
        vectorizer = SimpleTfidfVectorizer(max_features=max_features, min_df=min_df)

    vectorizer.fit(texts)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"  TF-IDF vectorizer saved → {save_path}")

    return vectorizer


def top_keywords(texts: pd.Series, vectorizer, top_n: int = 15) -> list:
    """Transform texts and return top-N keywords by mean TF-IDF score."""
    if texts.empty:
        return []

    matrix = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(matrix, "A1"):  # sparse matrix from sklearn
        mean_scores = matrix.mean(axis=0).A1
    else:  # dense numpy array
        mean_scores = matrix.mean(axis=0)

    top_idx = mean_scores.argsort()[::-1][:top_n]
    return [
        {"keyword": feature_names[i], "score": round(float(mean_scores[i]), 5)}
        for i in top_idx
    ]


def keywords_by_country(df: pd.DataFrame, vectorizer, top_n: int = 15) -> dict:
    result = {}
    for country, grp in df.groupby("country"):
        texts = build_text_field(grp)
        result[country] = top_keywords(texts, vectorizer, top_n)
    return result


def keywords_for_bursts(df: pd.DataFrame, burst_days: pd.DataFrame, vectorizer, top_n: int = 15) -> dict:
    burst_only = burst_days[burst_days["is_burst"]][["day", "country"]]
    if burst_only.empty:
        return {}
    merged = df.merge(burst_only, on=["day", "country"], how="inner")
    result = {}
    for country, grp in merged.groupby("country"):
        texts = build_text_field(grp)
        result[country] = top_keywords(texts, vectorizer, top_n)
    return result
