"""
embeddings.py
-------------
Local, free sentence-transformer embeddings + FAISS vector index.

Model : all-MiniLM-L6-v2  (384-dim, ~80 MB, runs fully on CPU)
Index : FAISS IndexFlatIP  (cosine similarity via L2-normalised dot product)
Cost  : $0 — runs entirely offline after first model download

Public API
----------
    embed_texts(texts)                            -> np.ndarray
    build_index(embeddings)                       -> faiss.IndexFlatIP
    save_index(index, metadata, idx_path, meta_path)
    load_index(idx_path, meta_path)               -> (index, metadata)
    search_index(index, metadata, query, top_k)   -> List[Dict]
    is_available()                                -> (bool, str)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional dependencies (graceful degradation) ──────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM        = 384    # all-MiniLM-L6-v2 output dimension
EMBED_BATCH      = 64     # sentences per encoding batch

# Module-level model singleton — loaded once, reused across calls
_model_singleton: Optional["SentenceTransformer"] = None


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL
# ═══════════════════════════════════════════════════════════════════════════════

def _require() -> None:
    """Raise ImportError with clear install instructions if deps are missing."""
    missing = []
    if not _HAS_ST:
        missing.append("sentence-transformers")
    if not _HAS_FAISS:
        missing.append("faiss-cpu")
    if missing:
        raise ImportError(
            f"Missing: {', '.join(missing)}. "
            f"Run: pip install {' '.join(missing)}"
        )


def _model() -> "SentenceTransformer":
    """Return the singleton embedding model (lazy-loaded)."""
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = SentenceTransformer(EMBED_MODEL_NAME)
    return _model_singleton


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def embed_texts(
    texts: List[str],
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Because vectors are L2-normalised, inner product == cosine similarity.
    This lets us use IndexFlatIP for fast cosine search with no approximation.

    Parameters
    ----------
    texts         : List of strings (may contain duplicates; all encoded)
    show_progress : Show tqdm progress bar (useful for large offline batches)

    Returns
    -------
    np.ndarray, shape (len(texts), EMBED_DIM), dtype float32
    Empty array of shape (0, EMBED_DIM) for empty input.
    """
    _require()
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    vecs = _model().encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,      # L2 normalise → cosine = dot product
    )
    return vecs.astype(np.float32)


def build_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """
    Build a FAISS inner-product (cosine) index from an embedding matrix.

    Parameters
    ----------
    embeddings : (N, EMBED_DIM) float32 array from embed_texts()

    Returns
    -------
    faiss.IndexFlatIP containing N vectors
    """
    _require()
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D float32 array")
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx


def save_index(
    index: "faiss.IndexFlatIP",
    metadata: List[Dict],
    index_path: str,
    meta_path: str,
) -> None:
    """
    Persist FAISS index + parallel metadata list to disk.

    Parameters
    ----------
    index      : FAISS index to save
    metadata   : List[Dict] — one entry per vector, must contain at least
                 {"text": str, "source_url": str, "date": str, "country": str}
    index_path : Path for the .faiss binary file
    meta_path  : Path for the metadata .pkl file
    """
    _require()
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


def load_index(
    index_path: str,
    meta_path: str,
) -> Tuple["faiss.IndexFlatIP", List[Dict]]:
    """
    Load FAISS index and metadata from disk.

    Raises FileNotFoundError if either file is missing.
    """
    _require()
    for p in (index_path, meta_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"RAG index file not found: {p}\n"
                "Build it first via the 'Show Evidence' button or "
                "run: python -m src.rag"
            )
    idx  = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return idx, meta


def search_index(
    index: "faiss.IndexFlatIP",
    metadata: List[Dict],
    query: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Find the top-k most semantically similar chunks to a query string.

    Parameters
    ----------
    index    : Loaded FAISS index
    metadata : Parallel metadata list (same order as index vectors)
    query    : Free-text query
    top_k    : How many results to return

    Returns
    -------
    List of metadata dicts with added "score" key (cosine similarity, 0–1).
    Sorted descending by score.
    """
    _require()
    q_vec = embed_texts([query])           # shape (1, EMBED_DIM)
    scores, indices = index.search(q_vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            entry = dict(metadata[idx])
            entry["score"] = round(float(score), 4)
            results.append(entry)

    return sorted(results, key=lambda x: x["score"], reverse=True)


def is_available() -> Tuple[bool, str]:
    """
    Check whether embedding features are available.

    Returns
    -------
    (True, "")              — ready
    (False, reason_string)  — missing dependency or config
    """
    if not _HAS_ST:
        return False, "sentence-transformers not installed (pip install sentence-transformers)"
    if not _HAS_FAISS:
        return False, "faiss-cpu not installed (pip install faiss-cpu)"
    return True, ""
