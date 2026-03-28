#!/usr/bin/env python3
"""Generate training labels using Gemini LLM (incremental, async, multi-item).

Samples random items from the dataset, classifies them via Gemini using
multi-item prompts (N items per request to save tokens) + async concurrency.
Appends results to a Parquet file every batch.
Supports resume: already-labeled item_ids are skipped automatically.

Usage:
    python scripts/generate_labels.py
    python scripts/generate_labels.py --sample-size 10000 --items-per-request 20 --concurrency 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_fallback import GeminiFallback
from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.generate_labels")


def load_existing_labels(output_path: Path) -> pd.DataFrame:
    """Load existing labels from Parquet if available."""
    if output_path.exists():
        return pd.read_parquet(output_path)
    return pd.DataFrame()


def append_to_parquet(new_df: pd.DataFrame, output_path: Path):
    """Append new labels to existing Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(output_path, index=False)
    return combined


def log_progress(batch_num: int, items_done: int, total: int, gemini: GeminiFallback, run_start: float):
    """Log detailed progress with LLM stats."""
    stats = gemini.stats
    elapsed = time.time() - run_start
    items_remaining = total - items_done
    throughput = items_done / elapsed if elapsed > 0 else 0
    eta_s = items_remaining / throughput if throughput > 0 else 0

    logger.info(
        "Batch %d | %d/%d (%.1f%%) | "
        "Elapsed: %.0fs | ETA: %.0fs | "
        "Throughput: %.1f items/s | "
        "Avg latency/req: %.2fs | "
        "Tokens: %d in + %d out = %d (saved ~%d) | "
        "Errors: %d (%.1f%%) | "
        "Hazmat: %d / Not: %d",
        batch_num, items_done, total, 100 * items_done / total,
        elapsed, eta_s,
        throughput,
        stats.avg_latency_s,
        stats.total_input_tokens, stats.total_output_tokens, stats.total_tokens,
        stats.tokens_saved_estimate,
        stats.failed_items, 100 * stats.error_rate,
        stats.hazmat_count, stats.not_hazmat_count,
    )


def log_final_summary(gemini: GeminiFallback, total_labeled: int, run_start: float, metrics_path: Path):
    """Log and persist final run summary."""
    stats = gemini.stats
    elapsed = time.time() - run_start
    items_this_run = stats.total_items
    throughput = items_this_run / elapsed if elapsed > 0 else 0

    summary = {
        "run_elapsed_s": round(elapsed, 1),
        "items_labeled_this_run": items_this_run,
        "total_labels_in_file": total_labeled,
        "throughput_items_per_s": round(throughput, 2),
        **stats.to_dict(),
    }

    logger.info("=" * 70)
    logger.info("LABELING SUMMARY")
    logger.info("=" * 70)
    logger.info("  Items labeled:      %d", items_this_run)
    logger.info("  Total in file:      %d", total_labeled)
    logger.info("  Elapsed:            %.1fs", elapsed)
    logger.info("  Throughput:         %.1f items/s", throughput)
    logger.info("  API requests:       %d (avg %.1f items/req)",
                stats.total_requests,
                stats.total_items / stats.total_requests if stats.total_requests > 0 else 0)
    logger.info("  Avg latency/req:    %.3fs", stats.avg_latency_s)
    logger.info("  Tokens (in):        %d", stats.total_input_tokens)
    logger.info("  Tokens (out):       %d", stats.total_output_tokens)
    logger.info("  Tokens (total):     %d", stats.total_tokens)
    logger.info("  Tokens saved:       ~%d (multi-item batching)", stats.tokens_saved_estimate)
    logger.info("  Errors:             %d (%.1f%%)", stats.failed_items, 100 * stats.error_rate)
    logger.info("  Hazmat:             %d", stats.hazmat_count)
    logger.info("  Not hazmat:         %d", stats.not_hazmat_count)
    logger.info("=" * 70)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Generate training labels via Gemini (async, multi-item)")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Total number of items to label")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Items per batch (saved to Parquet after each batch)")
    parser.add_argument("--items-per-request", type=int, default=5,
                        help="Items per API request (multi-item prompt)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests")
    parser.add_argument("--input", type=str, default="data/processed/items_raw.parquet")
    parser.add_argument("--output", type=str, default="data/processed/labels_llm.parquet")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metrics_path = Path("data/output/labeling_metrics.json")

    if not input_path.exists():
        logger.error("Input file not found: %s. Run collect_data.py first.", input_path)
        return

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d items from %s", len(df), input_path)

    # Load existing labels for resume
    existing = load_existing_labels(output_path)
    already_done = set(existing["item_id"].tolist()) if not existing.empty else set()
    logger.info("Existing labels: %d", len(already_done))

    # Random sample excluding already-labeled items
    available = df[~df["item_id"].isin(already_done)]
    n_to_sample = min(args.sample_size - len(already_done), len(available))

    if n_to_sample <= 0:
        logger.info("Already have %d labels (target: %d). Nothing to do.",
                     len(already_done), args.sample_size)
        return

    sample = available.sample(n=n_to_sample, random_state=42)
    logger.info("Sampled %d items to label (target total: %d)", len(sample), args.sample_size)

    # Prepare
    gemini = GeminiFallback(items_per_request=args.items_per_request)
    items = sample[["item_id", "title", "description"]].to_dict("records")
    merge_cols = ["item_id", "title", "description"]
    if "domain_id" in sample.columns:
        merge_cols.append("domain_id")
    if "category_id" in sample.columns:
        merge_cols.append("category_id")

    total_labeled = len(already_done)
    total_to_process = len(items)
    run_start = time.time()

    effective_requests = total_to_process / args.items_per_request
    logger.info(
        "Starting labeling: %d items | %d items/req | %d concurrent | ~%.0f API requests",
        total_to_process, args.items_per_request, args.concurrency, effective_requests,
    )

    # Process in batches, each batch uses async multi-item requests
    for batch_num, batch_start in enumerate(range(0, total_to_process, args.batch_size), start=1):
        batch_items = items[batch_start:batch_start + args.batch_size]

        results = gemini.classify_batch(batch_items, concurrency=args.concurrency)

        batch_df = pd.DataFrame(results)
        batch_df = batch_df.merge(sample[merge_cols], on="item_id", how="left")
        combined = append_to_parquet(batch_df, output_path)
        total_labeled = len(combined)

        items_done = min(batch_start + args.batch_size, total_to_process)
        log_progress(batch_num, items_done, total_to_process, gemini, run_start)

    log_final_summary(gemini, total_labeled, run_start, metrics_path)


if __name__ == "__main__":
    main()
