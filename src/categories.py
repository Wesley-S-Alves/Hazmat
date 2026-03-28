"""Category lookup table for Mercado Libre categories.

Fetches and caches category names for enriching the final dataset.
"""

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger("hazmat.categories")

BASE_URL = "https://api.mercadolibre.com"
CACHE_FILE = Path("data/processed/categories_lookup.json")


def load_cache() -> dict[str, str]:
    """Load cached category lookup table."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(lookup: dict[str, str]) -> None:
    """Save category lookup table to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)


def fetch_category_name(category_id: str) -> str:
    """Fetch category name from Mercado Libre API."""
    try:
        resp = requests.get(f"{BASE_URL}/categories/{category_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", category_id)
    except requests.RequestException as e:
        logger.warning("Failed to fetch category %s: %s", category_id, e)
        return category_id


def build_lookup(category_ids: list[str]) -> dict[str, str]:
    """Build a lookup table for all unique category IDs.

    Only fetches categories not already in cache.
    """
    lookup = load_cache()
    missing = [cid for cid in set(category_ids) if cid not in lookup]

    if missing:
        logger.info("Fetching %d new category names...", len(missing))
        for cid in missing:
            lookup[cid] = fetch_category_name(cid)
        save_cache(lookup)
        logger.info("Category lookup updated (%d total)", len(lookup))

    return lookup


def enrich_dataframe(df, lookup: dict[str, str] | None = None):
    """Add category_name column to a DataFrame with category_id."""
    if lookup is None:
        lookup = build_lookup(df["category_id"].dropna().unique().tolist())
    df["category_name"] = df["category_id"].map(lookup)
    return df
