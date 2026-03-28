#!/usr/bin/env python3
"""Collect ~100k products from Mercado Libre Catalog API.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --target 50000  # smaller test run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector import MeliCollector
from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.collect")


def main():
    parser = argparse.ArgumentParser(description="Collect products from Mercado Libre")
    parser.add_argument("--target", type=int, default=100_000, help="Target number of items")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    collector = MeliCollector(output_dir=Path(args.output))

    if not collector.access_token:
        logger.error(
            "No MELI_ACCESS_TOKEN found. Set token in .env for data collection."
        )
        return

    logger.info("Starting collection of ~%d products...", args.target)
    df = collector.collect_all(target_total=args.target)

    if df.empty:
        logger.error("No products collected!")
        return

    # Save combined dataset
    output_path = Path("data/processed/items_raw.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("Collection complete!")
    logger.info("  Total products: %d", len(df))
    logger.info("  Unique domains: %d", df["domain_id"].nunique())
    logger.info("  Products with description: %d", (df["description"].str.len() > 0).sum())
    logger.info("  Saved to: %s", output_path)


if __name__ == "__main__":
    main()
