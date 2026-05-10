"""
article_fetcher.py
------------------
Fetch and clean article text from GDELT source URLs.

Design
------
- Primary extractor : trafilatura  (best at main-content extraction)
- Fallback extractor: BeautifulSoup (paragraph-level heuristic)
- Hard timeout      : 8 seconds per request (never blocks the UI)
- Graceful failure  : returns "" on any error; caller decides what to do

Public API
----------
    fetch_article(url)                         -> str
    fetch_burst_articles(df, date_str, country, max_articles) -> List[Dict]
    get_burst_source_urls(df, date_str, country, n)           -> List[str]
"""

from __future__ import annotations

import re
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

# ── Optional heavy extractors ─────────────────────────────────────────────────
try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

# ── Constants ─────────────────────────────────────────────────────────────────
_TIMEOUT      = 8          # seconds per HTTP request
_MAX_CHARS    = 8_000      # truncate article text at this length
_MIN_CHARS    = 150        # discard articles shorter than this
_CACHE_DIR    = Path("data/article_cache")
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GDELTResearch/1.0; "
        "+https://github.com/gdelt-event-intelligence)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Domains that consistently block or return garbage — skip immediately
_SKIP_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "tiktok.com", "reddit.com",
    "wsj.com", "ft.com", "bloomberg.com",         # paywalls
    "nytimes.com", "washingtonpost.com",            # paywalls
    "t.co", "bit.ly", "tinyurl.com",               # shorteners
}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _url_to_cache_key(url: str) -> str:
    """Deterministic short hash for a URL → used as cache filename."""
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _is_skippable(url: str) -> bool:
    """Return True if we should not even attempt to fetch this URL."""
    try:
        domain = urlparse(url).netloc.lstrip("www.")
        return any(domain == d or domain.endswith("." + d) for d in _SKIP_DOMAINS)
    except Exception:
        return True


def _extract_with_trafilatura(html: str, url: str) -> str:
    """Use trafilatura for main-content extraction (best quality)."""
    if not _HAS_TRAFILATURA:
        return ""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        return (text or "").strip()
    except Exception:
        return ""


def _extract_with_bs4(html: str) -> str:
    """Fallback: grab all <p> tags, join them."""
    if not _HAS_BS4:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove navigation, headers, footers, scripts, styles
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "button", "noscript"]):
            tag.decompose()
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        return " ".join(p for p in paragraphs if len(p) > 40)
    except Exception:
        return ""


def _clean_text(raw: str) -> str:
    """Normalise whitespace and remove non-printable characters."""
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    raw = re.sub(r" {2,}", " ", raw)
    raw = re.sub(r"[^\x20-\x7E]", " ", raw)          # keep ASCII printable
    return raw.strip()[:_MAX_CHARS]


def _load_from_cache(url: str) -> Optional[str]:
    """Return cached article text or None."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / (_url_to_cache_key(url) + ".txt")
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def _save_to_cache(url: str, text: str) -> None:
    """Save article text to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / (_url_to_cache_key(url) + ".txt")
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_article(url: str, use_cache: bool = True) -> str:
    """
    Fetch and extract readable text from a URL.

    Returns
    -------
    Clean article text (possibly empty string on failure).
    Never raises — all exceptions are caught and logged.

    Strategy
    --------
    1. Skip known-bad domains instantly.
    2. Check disk cache (avoids re-fetching during demo).
    3. HTTP GET with timeout + realistic headers.
    4. Try trafilatura first, fall back to BeautifulSoup.
    5. Discard if result < MIN_CHARS.
    6. Cache successful results.
    """
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()

    if _is_skippable(url):
        return ""

    # Cache check
    if use_cache:
        cached = _load_from_cache(url)
        if cached is not None:
            return cached

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
        if resp.status_code != 200:
            return ""
        html = resp.text
    except Exception as exc:
        log.debug("Fetch failed (%s): %s", url[:60], exc)
        return ""

    # Extraction
    text = _extract_with_trafilatura(html, url)
    if not text:
        text = _extract_with_bs4(html)

    text = _clean_text(text)

    if len(text) < _MIN_CHARS:
        return ""

    if use_cache:
        _save_to_cache(url, text)

    return text


def get_burst_source_urls(
    df,
    date_str: str,
    country: str,
    n: int = 20,
) -> List[str]:
    """
    Return up to `n` unique GDELT source URLs for events on a burst day.

    Prioritises events with:
    - Higher NumMentions (more coverage)
    - Conflict events (QuadClass 3 or 4) over cooperation

    Parameters
    ----------
    df       : Full events DataFrame (must have event_date, country, SOURCEURL)
    date_str : "YYYY-MM-DD" burst date
    country  : Country name, e.g. "Iran"
    n        : Max URLs to return

    Returns
    -------
    List of URL strings (deduplicated, skippable domains removed)
    """
    import pandas as pd

    if "SOURCEURL" not in df.columns:
        return []

    target_date = pd.Timestamp(date_str)
    day_mask = (
        (df["day"] == target_date) &
        (df["country"] == country) &
        df["SOURCEURL"].notna()
    )
    day_df = df[day_mask].copy()

    if day_df.empty:
        return []

    # Sort: conflict events first, then by mention count
    if "QuadClass" in day_df.columns:
        day_df["_conflict_flag"] = (day_df["QuadClass"] >= 3).astype(int)
        day_df = day_df.sort_values(
            ["_conflict_flag", "NumMentions"],
            ascending=[False, False],
        )

    urls = day_df["SOURCEURL"].dropna().unique().tolist()
    # Filter skippable domains
    urls = [u for u in urls if not _is_skippable(u)]
    return urls[:n]


def fetch_burst_articles(
    df,
    date_str: str,
    country: str,
    max_articles: int = 10,
    delay_sec: float = 0.3,
) -> List[Dict]:
    """
    Fetch article text for a burst day.

    Parameters
    ----------
    df           : Full events DataFrame
    date_str     : "YYYY-MM-DD" burst date
    country      : Country name
    max_articles : Max articles to attempt
    delay_sec    : Polite delay between requests

    Returns
    -------
    List of dicts:
        {
          "url"     : str,
          "text"    : str,      # extracted article text
          "country" : str,
          "date"    : str,
          "domain"  : str,
        }
    Only entries with non-empty text are returned.
    """
    urls = get_burst_source_urls(df, date_str, country, n=max_articles * 3)
    results = []
    attempted = 0

    for url in urls:
        if attempted >= max_articles:
            break
        text = fetch_article(url)
        attempted += 1
        if text:
            domain = urlparse(url).netloc.lstrip("www.")
            results.append({
                "url"    : url,
                "text"   : text,
                "country": country,
                "date"   : date_str,
                "domain" : domain,
            })
        if delay_sec > 0:
            time.sleep(delay_sec)

    log.info(
        "fetch_burst_articles: %d/%d URLs yielded text for %s %s",
        len(results), attempted, country, date_str,
    )
    return results
