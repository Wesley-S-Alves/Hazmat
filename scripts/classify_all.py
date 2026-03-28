#!/usr/bin/env python3
"""Classify all 100k items using the ensemble pipeline.

Usage:
    python scripts/classify_all.py
    python scripts/classify_all.py --no-llm          # skip LLM fallback
    python scripts/classify_all.py --threshold 0.7    # custom confidence threshold
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability import PipelineObserver, setup_logging
from src.pipeline import HazmatPipeline

setup_logging()
logger = logging.getLogger("hazmat.classify_all")


def main():
    parser = argparse.ArgumentParser(description="Classify all items with ensemble pipeline")
    parser.add_argument("--input", type=str, default="data/processed/items_raw.parquet")
    parser.add_argument("--output", type=str, default="data/output/hazmat_classified_100k.csv")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM fallback")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold for LLM fallback (default: from model config)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input not found: %s. Run collect_data.py first.", input_path)
        return

    # Load data
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d items", len(df))

    # Initialize pipeline
    observer = PipelineObserver(output_dir=output_path.parent)
    pipeline = HazmatPipeline(observer=observer)

    # Load trained models
    models_dir = Path("models")
    try:
        pipeline.load_models(models_dir)
        logger.info("Ensemble models loaded from %s", models_dir)
    except Exception as e:
        logger.error("Could not load models: %s. Run train_model.py first.", e)
        return

    # Confidence threshold for LLM fallback (separate from decision threshold)
    # Decision threshold (0.315) = when to predict hazmat vs non-hazmat
    # Confidence threshold (0.8) = when ML is uncertain enough to ask LLM
    if args.threshold is not None:
        pipeline.confidence_threshold = args.threshold
        logger.info("Using custom confidence threshold for LLM: %.3f", args.threshold)
    else:
        pipeline.confidence_threshold = 0.8
        logger.info("Using confidence threshold for LLM fallback: %.3f", pipeline.confidence_threshold)

    if args.no_llm:
        logger.info("LLM fallback disabled.")
        pipeline.llm_fallback = None

    # Classify
    df = pipeline.classify_batch(df, use_llm=not args.no_llm)

    # Flag items for human review:
    # Flow: ML → low confidence → LLM → still low confidence → human
    # Only items that already went through LLM and STILL have low confidence
    df["needs_human_review"] = False

    if "source_layer" in df.columns:
        # LLM classified but confidence < 0.8 → human must verify
        llm_low_conf = (df["source_layer"] == "llm") & (df["confidence_score"] < 0.8)
        df.loc[llm_low_conf, "needs_human_review"] = True

    n_review = df["needs_human_review"].sum()
    logger.info("Items flagged for human review: %d (%.1f%%)", n_review, 100 * n_review / len(df))

    # Select output columns
    output_cols = [
        "item_id", "title", "description", "is_hazmat", "reason",
        "domain_id", "confidence_score", "source_layer",
        "ensemble_proba", "needs_human_review",
    ]
    # Add per-model probabilities if available
    for col in df.columns:
        if col.startswith("proba_"):
            output_cols.append(col)

    existing_cols = [c for c in output_cols if c in df.columns]
    df_out = df[existing_cols]

    # Save CSV + Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    parquet_path = output_path.with_suffix(".parquet")
    df_out.to_parquet(parquet_path, index=False)
    logger.info("Saved: %s + %s", output_path, parquet_path)

    # Save metrics
    observer.save_metrics()

    # Summary
    logger.info("=" * 60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("  Total items:       %d", len(df_out))
    hazmat_count = df_out["is_hazmat"].sum()
    logger.info("  Hazmat:            %d (%.1f%%)", hazmat_count, 100 * hazmat_count / len(df_out))
    logger.info("  By source:")
    if "source_layer" in df_out.columns:
        for layer, count in df_out["source_layer"].value_counts().items():
            logger.info("    %s: %d (%.1f%%)", layer, count, 100 * count / len(df_out))
    logger.info("  Human review:      %d (%.1f%%)", n_review, 100 * n_review / len(df_out))
    logger.info("  Output: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
