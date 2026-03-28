#!/usr/bin/env python3
"""Real-time classification test with full observability.

Simulates production: fetches items from ML API or uses test samples,
classifies them one-by-one through the pipeline, and tracks detailed
performance metrics.

Usage:
    # Interactive mode (manual input)
    python scripts/realtime_test.py --mode interactive

    # Simulate stream from parquet (random items, one at a time)
    python scripts/realtime_test.py --mode stream --n-items 50

    # Fetch fresh items from ML API
    python scripts/realtime_test.py --mode api --query "bateria litio" --n-items 20
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability import setup_logging

# Silence noisy loggers
for noisy in ["google_genai", "httpx", "httpcore", "urllib3", "sentence_transformers"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger("hazmat.realtime")


@dataclass
class RealtimeMetrics:
    """Track real-time classification metrics."""

    items_total: int = 0
    items_hazmat: int = 0
    items_not_hazmat: int = 0
    items_ml: int = 0
    items_llm: int = 0
    items_human_review: int = 0

    # Latency tracking (ms)
    latencies_embedding: list = field(default_factory=list)
    latencies_ml: list = field(default_factory=list)
    latencies_llm: list = field(default_factory=list)
    latencies_total: list = field(default_factory=list)

    # Confidence tracking
    confidences: list = field(default_factory=list)
    confidences_ml: list = field(default_factory=list)
    confidences_llm: list = field(default_factory=list)

    # Token tracking (LLM only)
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    # Cost estimation (Gemini Flash pricing)
    cost_per_1k_input_tokens: float = 0.00001875  # $0.075 per 1M
    cost_per_1k_output_tokens: float = 0.000075    # $0.30 per 1M

    def record(self, result: dict):
        self.items_total += 1
        if result["is_hazmat"]:
            self.items_hazmat += 1
        else:
            self.items_not_hazmat += 1

        source = result.get("source_layer", "ml")
        conf = result.get("confidence_score", 0)
        self.confidences.append(conf)

        if source == "ml":
            self.items_ml += 1
            self.confidences_ml.append(conf)
        elif source == "llm":
            self.items_llm += 1
            self.confidences_llm.append(conf)
            self.llm_input_tokens += result.get("llm_input_tokens", 0)
            self.llm_output_tokens += result.get("llm_output_tokens", 0)

        if result.get("needs_human_review", False):
            self.items_human_review += 1

        # Latencies
        if "latency_embedding_ms" in result:
            self.latencies_embedding.append(result["latency_embedding_ms"])
        if "latency_ml_ms" in result:
            self.latencies_ml.append(result["latency_ml_ms"])
        if "latency_llm_ms" in result:
            self.latencies_llm.append(result["latency_llm_ms"])
        if "latency_total_ms" in result:
            self.latencies_total.append(result["latency_total_ms"])

    @property
    def estimated_cost(self) -> float:
        input_cost = (self.llm_input_tokens / 1000) * self.cost_per_1k_input_tokens
        output_cost = (self.llm_output_tokens / 1000) * self.cost_per_1k_output_tokens
        return input_cost + output_cost

    def _percentiles(self, values: list) -> dict:
        if not values:
            return {"p50": 0, "p90": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "avg": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def summary(self) -> dict:
        return {
            "items": {
                "total": self.items_total,
                "hazmat": self.items_hazmat,
                "not_hazmat": self.items_not_hazmat,
                "hazmat_rate": round(self.items_hazmat / self.items_total, 3) if self.items_total > 0 else 0,
            },
            "routing": {
                "ml": self.items_ml,
                "llm": self.items_llm,
                "ml_pct": round(100 * self.items_ml / self.items_total, 1) if self.items_total > 0 else 0,
                "llm_pct": round(100 * self.items_llm / self.items_total, 1) if self.items_total > 0 else 0,
                "human_review": self.items_human_review,
                "human_review_pct": round(100 * self.items_human_review / self.items_total, 1) if self.items_total > 0 else 0,
            },
            "latency_ms": {
                "total": self._percentiles(self.latencies_total),
                "embedding": self._percentiles(self.latencies_embedding),
                "ml_inference": self._percentiles(self.latencies_ml),
                "llm_inference": self._percentiles(self.latencies_llm),
            },
            "confidence": {
                "all": self._percentiles(self.confidences),
                "ml_only": self._percentiles(self.confidences_ml),
                "llm_only": self._percentiles(self.confidences_llm),
            },
            "cost": {
                "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens,
                "llm_total_tokens": self.llm_input_tokens + self.llm_output_tokens,
                "estimated_cost_usd": round(self.estimated_cost, 6),
            },
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 70)
        print("REAL-TIME CLASSIFICATION METRICS")
        print("=" * 70)

        print(f"\n  Items classified:   {s['items']['total']}")
        print(f"  Hazmat:             {s['items']['hazmat']} ({s['items']['hazmat_rate']*100:.1f}%)")
        print(f"  Non-hazmat:         {s['items']['not_hazmat']}")

        print(f"\n  Routing:")
        print(f"    ML resolved:      {s['routing']['ml']} ({s['routing']['ml_pct']}%)")
        print(f"    LLM fallback:     {s['routing']['llm']} ({s['routing']['llm_pct']}%)")
        print(f"    Human review:     {s['routing']['human_review']} ({s['routing']['human_review_pct']}%)")

        print(f"\n  Latency (ms):")
        for stage, stats in s["latency_ms"].items():
            if stats["avg"] > 0:
                print(f"    {stage:18s}  avg={stats['avg']:7.1f}  p50={stats['p50']:7.1f}  p90={stats['p90']:7.1f}  p99={stats['p99']:7.1f}")

        print(f"\n  Confidence:")
        for scope, stats in s["confidence"].items():
            if stats["avg"] > 0:
                print(f"    {scope:18s}  avg={stats['avg']:.3f}  p50={stats['p50']:.3f}  min={stats['min']:.3f}")

        print(f"\n  LLM Cost:")
        print(f"    Tokens (in/out):  {s['cost']['llm_input_tokens']} / {s['cost']['llm_output_tokens']}")
        print(f"    Estimated cost:   ${s['cost']['estimated_cost_usd']:.6f}")

        print("=" * 70)


def classify_single_with_metrics(pipeline, title: str, description: str = "",
                                  category_id: str = "") -> dict:
    """Classify a single item and return result with latency breakdown."""
    from src.features import FeatureBuilder

    result = {}
    total_start = time.time()

    # Step 1: Embedding
    emb_start = time.time()
    row_df = pd.DataFrame([{
        "title": title,
        "description": description,
        "category_id": category_id,
        "domain_id": category_id,
    }])
    X = pipeline.feature_builder.transform(row_df)
    emb_ms = (time.time() - emb_start) * 1000

    # Step 2: ML prediction
    ml_start = time.time()
    detailed = pipeline.ml_classifier.predict_detailed(X)
    ml_ms = (time.time() - ml_start) * 1000

    pred = detailed["predictions"][0]
    conf = detailed["confidence"][0]
    proba = detailed["ensemble_proba"][0]

    result["is_hazmat"] = bool(pred)
    result["confidence_score"] = float(conf)
    result["ensemble_proba"] = float(proba)
    result["source_layer"] = "ml"
    result["reason"] = f"Ensemble ML (confidence: {conf:.2f})"
    result["latency_embedding_ms"] = emb_ms
    result["latency_ml_ms"] = ml_ms
    result["needs_human_review"] = False

    # Per-model breakdown
    for model_name, model_proba in detailed["per_model"].items():
        result[f"proba_{model_name}"] = float(model_proba[0])

    # Step 3: LLM fallback if low confidence
    if conf < pipeline.confidence_threshold and pipeline.llm_fallback:
        llm_start = time.time()
        llm_result = pipeline.llm_fallback.classify(title, description)
        llm_ms = (time.time() - llm_start) * 1000

        result["is_hazmat"] = llm_result["is_hazmat"]
        result["reason"] = llm_result["reason"]
        result["confidence_score"] = llm_result.get("confidence", 0.85)
        result["source_layer"] = "llm"
        result["latency_llm_ms"] = llm_ms
        result["llm_input_tokens"] = llm_result.get("input_tokens", 0)
        result["llm_output_tokens"] = llm_result.get("output_tokens", 0)

        if llm_result.get("confidence", 1.0) < 0.7:
            result["needs_human_review"] = True

    total_ms = (time.time() - total_start) * 1000
    result["latency_total_ms"] = total_ms

    return result


def print_item_result(item: dict, result: dict, index: int = 0):
    """Pretty print a single classification result."""
    hazmat_icon = "!!" if result["is_hazmat"] else "ok"
    source = result["source_layer"].upper()
    conf = result["confidence_score"]
    total_ms = result["latency_total_ms"]
    review = " [HUMAN REVIEW]" if result.get("needs_human_review") else ""

    print(f"\n  [{index:3d}] [{hazmat_icon}] {item.get('title', '')[:60]}")
    print(f"        Source: {source} | Confidence: {conf:.3f} | Latency: {total_ms:.1f}ms{review}")
    print(f"        Reason: {result['reason']}")

    if "proba_xgboost" in result:
        print(f"        Models: XGB={result['proba_xgboost']:.3f} "
              f"LGB={result['proba_lightgbm']:.3f} "
              f"RF={result['proba_random_forest']:.3f} "
              f"Ensemble={result['ensemble_proba']:.3f}")


def mode_interactive(pipeline, metrics: RealtimeMetrics):
    """Interactive mode: user types title + description."""
    print("\n" + "=" * 70)
    print("INTERACTIVE HAZMAT CLASSIFIER")
    print("Type product title (or 'quit' to exit)")
    print("Optionally add description after '|'")
    print("Example: Bateria Litio 18650 | Bateria recarregavel 3.7V 3000mAh")
    print("=" * 70)

    idx = 0
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        parts = user_input.split("|", 1)
        title = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""

        result = classify_single_with_metrics(pipeline, title, description)
        metrics.record(result)
        idx += 1

        print_item_result({"title": title}, result, idx)

    metrics.print_summary()


def mode_stream(pipeline, metrics: RealtimeMetrics, n_items: int = 50):
    """Stream mode: random items from parquet, one at a time."""
    input_path = Path("data/processed/items_raw.parquet")
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return

    df = pd.read_parquet(input_path)
    sample = df.sample(n=min(n_items, len(df)), random_state=None)  # random each run

    print(f"\n  Streaming {len(sample)} random items...\n")

    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        title = row.get("title", "")
        description = row.get("description", "")
        category_id = row.get("domain_id", row.get("category_id", ""))

        result = classify_single_with_metrics(pipeline, title, description, category_id)
        metrics.record(result)

        print_item_result({"title": title}, result, idx)

    metrics.print_summary()


def mode_api(pipeline, metrics: RealtimeMetrics, query: str = "bateria litio", n_items: int = 20):
    """API mode: fetch fresh items from ML API and classify."""
    try:
        from src.collector import MeliCollector
    except ImportError:
        logger.error("MeliCollector not available")
        return

    collector = MeliCollector()
    print(f"\n  Fetching '{query}' from ML API...")

    items = collector.search(query, limit=n_items)
    if not items:
        logger.error("No items returned from API")
        return

    print(f"  Got {len(items)} items. Classifying...\n")

    for idx, item in enumerate(items, 1):
        title = item.get("title", "")
        description = item.get("description", "")
        category_id = item.get("domain_id", item.get("category_id", ""))

        result = classify_single_with_metrics(pipeline, title, description, category_id)
        metrics.record(result)

        print_item_result({"title": title}, result, idx)

    metrics.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Real-time classification test with o11y")
    parser.add_argument("--mode", choices=["interactive", "stream", "api"],
                        default="stream", help="Test mode")
    parser.add_argument("--n-items", type=int, default=50, help="Items to test (stream/api)")
    parser.add_argument("--query", type=str, default="bateria litio", help="API search query")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback")
    parser.add_argument("--output", type=str, default="data/output/realtime_metrics.json",
                        help="Save metrics to file")
    args = parser.parse_args()

    # Load pipeline
    from src.pipeline import HazmatPipeline

    logger.info("Loading pipeline...")
    pipeline = HazmatPipeline()
    pipeline.load_models()

    if args.no_llm:
        pipeline.llm_fallback = None
        logger.info("LLM fallback disabled")

    # Use model's optimized threshold
    if pipeline.ml_classifier:
        pipeline.confidence_threshold = pipeline.ml_classifier.threshold
    logger.info("Confidence threshold: %.3f", pipeline.confidence_threshold)

    logger.info("Pipeline ready. Starting %s mode...", args.mode)

    metrics = RealtimeMetrics()

    if args.mode == "interactive":
        mode_interactive(pipeline, metrics)
    elif args.mode == "stream":
        mode_stream(pipeline, metrics, args.n_items)
    elif args.mode == "api":
        mode_api(pipeline, metrics, args.query, args.n_items)

    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics.summary(), f, indent=2)
    logger.info("Metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
