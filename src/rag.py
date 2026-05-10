
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

from .article_fetcher import fetch_burst_articles
from .embeddings import embed_texts, build_index, save_index, load_index, search_index, is_available


_RAG_DIR = Path("data/rag")


_CHUNK_CHARS   = 500    
_CHUNK_OVERLAP = 80     
_MIN_CHUNK     = 80     


def _burst_key(date_str: str, country: str) -> str:
    """Filesystem-safe key for a burst. E.g. '2026-01-12_Iran'"""
    return f"{date_str}_{country.replace(' ', '_')}"


def _index_paths(date_str: str, country: str) -> Tuple[str, str]:
    """Return (faiss_path, meta_path) for a burst."""
    key = _burst_key(date_str, country)
    _RAG_DIR.mkdir(parents=True, exist_ok=True)
    return str(_RAG_DIR / f"{key}.faiss"), str(_RAG_DIR / f"{key}_meta.pkl")




def chunk_text(
    text: str,
    chunk_size: int = _CHUNK_CHARS,
    overlap: int = _CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping fixed-size character chunks.

    Attempts to break at sentence boundaries ('. ') to avoid
    cutting mid-sentence. Falls back to hard split if no boundary found.

    Parameters
    ----------
    text       : Input text (any length)
    chunk_size : Target chunk length in characters
    overlap    : Characters shared between consecutive chunks

    Returns
    -------
    List of non-empty chunk strings, each >= _MIN_CHUNK chars.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        
        if end < n:
            boundary = text.rfind(". ", start + chunk_size - 120, end)
            if boundary != -1:
                end = boundary + 2     # include the ". "

        chunk = text[start:end].strip()
        if len(chunk) >= _MIN_CHUNK:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        start = next_start

    return chunks


def _chunks_to_metadata(
    chunks: List[str],
    article: Dict,
) -> List[Dict]:
    """Attach article metadata to each chunk."""
    return [
        {
            "text"       : chunk,
            "source_url" : article.get("url", ""),
            "domain"     : article.get("domain", ""),
            "date"       : article.get("date", ""),
            "country"    : article.get("country", ""),
            "chunk_idx"  : i,
        }
        for i, chunk in enumerate(chunks)
    ]




def burst_index_exists(date_str: str, country: str) -> bool:
    """Check whether a FAISS index already exists on disk for this burst."""
    idx_p, meta_p = _index_paths(date_str, country)
    return Path(idx_p).exists() and Path(meta_p).exists()


def build_burst_rag(
    df,
    date_str: str,
    country: str,
    max_articles: int = 10,
    force_rebuild: bool = False,
    progress_callback=None,
) -> Tuple[Optional[object], List[Dict], List[Dict]]:
    """
    Build (or load from cache) a FAISS RAG index for a burst day.

    Parameters
    ----------
    df              : Full events DataFrame
    date_str        : "YYYY-MM-DD" burst date
    country         : Country name
    max_articles    : Max articles to fetch
    force_rebuild   : Ignore disk cache and rebuild
    progress_callback: Optional callable(message: str) for UI status updates

    Returns
    -------
    (faiss_index, chunk_metadata_list, raw_articles_list)
    Returns (None, [], []) if embeddings are not available or no text fetched.
    """
    ok, reason = is_available()
    if not ok:
        log.warning("Embeddings unavailable: %s", reason)
        return None, [], []

    idx_path, meta_path = _index_paths(date_str, country)

    # ── Load from cache if available ─────────────────────────────────────────
    if not force_rebuild and burst_index_exists(date_str, country):
        try:
            idx, meta = load_index(idx_path, meta_path)
            log.info("Loaded RAG index from cache: %s %s", country, date_str)
            return idx, meta, []         # articles not cached separately
        except Exception as e:
            log.warning("Cache load failed (%s), rebuilding: %s", meta_path, e)

    # ── Fetch articles ────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(f"Fetching articles for {country} on {date_str}…")

    articles = fetch_burst_articles(
        df, date_str, country, max_articles=max_articles
    )

    if not articles:
        log.info("No articles fetched for %s %s", country, date_str)
        return None, [], []

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks: List[str] = []
    all_meta:   List[Dict] = []

    for article in articles:
        chunks = chunk_text(article["text"])
        meta   = _chunks_to_metadata(chunks, article)
        all_chunks.extend(chunks)
        all_meta.extend(meta)

    if not all_chunks:
        return None, [], articles

    # ── Embed ─────────────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(f"Embedding {len(all_chunks)} text chunks…")

    embeddings = embed_texts(all_chunks)

    # ── Index ─────────────────────────────────────────────────────────────────
    index = build_index(embeddings)
    save_index(index, all_meta, idx_path, meta_path)

    if progress_callback:
        progress_callback(
            f"RAG index built: {len(articles)} articles, "
            f"{len(all_chunks)} chunks."
        )

    return index, all_meta, articles


def retrieve_for_spike(
    query: str,
    date_str: str,
    country: str,
    top_k: int = 4,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant article chunks for a query.

    Requires that build_burst_rag() has been called first
    (index is loaded from disk automatically).

    Parameters
    ----------
    query    : Free-text query string
    date_str : Burst date
    country  : Country name
    top_k    : Number of chunks to return

    Returns
    -------
    List of metadata dicts with "text", "source_url", "domain", "score".
    Returns [] if index not found or embeddings unavailable.
    """
    ok, _ = is_available()
    if not ok:
        return []

    if not burst_index_exists(date_str, country):
        return []

    idx_path, meta_path = _index_paths(date_str, country)
    try:
        index, metadata = load_index(idx_path, meta_path)
    except FileNotFoundError:
        return []

    return search_index(index, metadata, query, top_k=top_k)


def format_rag_context(
    chunks: List[Dict],
    max_chars: int = 1200,
) -> str:
    """
    Format retrieved chunks as a readable context block.

    Parameters
    ----------
    chunks    : Output of retrieve_for_spike() or search_index()
    max_chars : Total character limit for the context block

    Returns
    -------
    Multi-line string ready to inject into a prompt or display.
    Example:
        [Source: reuters.com | Score: 0.87]
        "Tensions rose sharply as Iran announced..."

        [Source: aljazeera.com | Score: 0.81]
        "The UN Security Council convened..."
    """
    if not chunks:
        return ""

    lines = []
    total = 0

    for chunk in chunks:
        domain = chunk.get("domain") or chunk.get("source_url", "unknown")[:40]
        score  = chunk.get("score", 0.0)
        text   = chunk.get("text", "").strip()

        entry = f'[Source: {domain} | Relevance: {score:.2f}]\n"{text}"\n'
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)

    return "\n".join(lines)
