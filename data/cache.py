"""
File-based JSON cache with TTL expiry.
All API calls route through here to avoid redundant fetches and protect
the Odds API 500-credit monthly limit.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from config import CACHE_DIR, CACHE_TTL_SECONDS


def _key_to_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(CACHE_DIR) / f"{digest}.json"


def cache_get(key: str, ttl: int = CACHE_TTL_SECONDS):
    """Return cached data if it exists and is fresh; otherwise None."""
    path = _key_to_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            entry = json.load(fh)
        if time.time() - entry["ts"] > ttl:
            return None
        return entry["data"]
    except Exception:
        return None


def cache_set(key: str, data) -> None:
    """Persist data to the cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _key_to_path(key)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"ts": time.time(), "data": data}, fh)


def cached_request(session, url: str, params: dict = None,
                   ttl: int = CACHE_TTL_SECONDS):
    """
    Perform a GET request through the cache.
    Returns parsed JSON dict/list.
    """
    import urllib.parse
    cache_key = url + ("?" + urllib.parse.urlencode(sorted((params or {}).items())) if params else "")
    hit = cache_get(cache_key, ttl)
    if hit is not None:
        return hit
    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    cache_set(cache_key, data)
    return data


def cache_clear_all() -> None:
    cache_dir = Path(CACHE_DIR)
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            f.unlink()
