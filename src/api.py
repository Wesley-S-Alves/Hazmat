"""Hazmat Classifier — FastAPI serving layer.

Ensemble ML classification with optional LLM fallback for low-confidence items.
Port 8080 (8000 is occupied by Splunk).
"""

from src.compat import configure_omp

configure_omp()

import logging
import os
import platform
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src.canary import CanaryRouter
from src.observability import PipelineObserver, setup_logging
from src.pipeline import HazmatPipeline
from src.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)

logger = logging.getLogger("hazmat.api")

# ── Constants ───────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
CANARY_DIR = Path(os.getenv("CANARY_MODELS_DIR", "models_canary"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
PORT = int(os.getenv("PORT", "8080"))
CANARY_PCT = float(os.getenv("CANARY_PCT", "0.10"))

# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, clean up on shutdown."""
    setup_logging()
    startup_time = time.time()
    app.state.startup_time = startup_time
    app.state.models_loaded = False

    pipeline = HazmatPipeline()
    observer = pipeline.observer

    # Attempt to load ML models
    try:
        pipeline.load_models(MODELS_DIR)
        # Warmup: force embedding model load now (not on first request)
        if pipeline.feature_builder:
            _ = pipeline.feature_builder.embedding_model
            logger.info("Embedding model warmed up on %s", pipeline.feature_builder.device)
        app.state.models_loaded = True
        logger.info(
            "Models loaded from %s | version=%s | device=%s",
            MODELS_DIR,
            MODEL_VERSION,
            _detect_device(),
        )
    except Exception:
        logger.warning(
            "ML models not found at %s — running in LLM-only mode.",
            MODELS_DIR,
            exc_info=True,
        )

    # Check LLM availability
    has_llm = bool(os.getenv("GEMINI_API_KEY"))
    if not has_llm:
        logger.info("GEMINI_API_KEY not set — LLM fallback disabled, ML-only mode.")

    app.state.pipeline = pipeline
    app.state.observer = observer

    # Canary router (optional — only active if canary model dir exists)
    canary_router = CanaryRouter(
        production_dir=MODELS_DIR,
        canary_dir=CANARY_DIR if CANARY_DIR.exists() else None,
        canary_pct=CANARY_PCT,
    )
    try:
        canary_router.load()
    except Exception:
        logger.info("Canary router: production-only mode (no canary model).")
    app.state.canary_router = canary_router

    logger.info(
        "Hazmat API ready on port %d | models_loaded=%s | llm_available=%s | canary=%s",
        PORT,
        app.state.models_loaded,
        has_llm,
        CANARY_DIR.exists(),
    )

    yield  # ── app is running ──

    logger.info("Shutting down Hazmat API.")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hazmat Classifier API",
    description="Ensemble ML + LLM fallback for hazardous-materials classification.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _detect_device() -> str:
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        return "apple-silicon"
    return "cpu"


def _get_pipeline() -> HazmatPipeline:
    pipeline: HazmatPipeline = app.state.pipeline
    if not pipeline.ml_classifier and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: no ML models loaded and no LLM key configured.",
        )
    return pipeline


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.post("/classify", response_model=ClassifyResponse)
async def classify_single(req: ClassifyRequest) -> ClassifyResponse:
    """Classify a single product as hazmat or not."""
    try:
        pipeline = _get_pipeline()
        result = pipeline.classify_single(
            title=req.title,
            description=req.description,
            category_id=req.category_id,
        )
        return ClassifyResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /classify")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(req: BatchClassifyRequest) -> BatchClassifyResponse:
    """Classify a batch of products (up to 100)."""
    try:
        pipeline = _get_pipeline()
        use_llm = bool(os.getenv("GEMINI_API_KEY"))

        df = pd.DataFrame([item.model_dump() for item in req.items])
        df["item_id"] = [f"api-{i}" for i in range(len(df))]
        df["domain_id"] = df["category_id"]

        result_df = pipeline.classify_batch(df, use_llm=use_llm)

        # Build per-item responses
        results: list[ClassifyResponse] = []
        for _, row in result_df.iterrows():
            per_model = {
                col.replace("proba_", ""): float(row[col])
                for col in result_df.columns
                if col.startswith("proba_")
            } or None

            results.append(
                ClassifyResponse(
                    is_hazmat=bool(row["is_hazmat"]),
                    confidence_score=float(row["confidence_score"]),
                    source_layer=str(row["source_layer"]),
                    reason=str(row["reason"]),
                    ensemble_proba=float(row["ensemble_proba"])
                    if "ensemble_proba" in row and pd.notna(row["ensemble_proba"])
                    else None,
                    per_model=per_model,
                )
            )

        # Summary stats
        total = len(results)
        hazmat_count = sum(1 for r in results if r.is_hazmat)
        ml_count = sum(1 for r in results if r.source_layer == "ml")
        llm_count = sum(1 for r in results if r.source_layer == "llm")

        return BatchClassifyResponse(
            results=results,
            summary={
                "total": total,
                "hazmat": hazmat_count,
                "ml_resolved": ml_count,
                "llm_fallback": llm_count,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /classify/batch")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check."""
    uptime = time.time() - app.state.startup_time
    return HealthResponse(
        status="ok" if app.state.models_loaded else "degraded",
        models_loaded=app.state.models_loaded,
        model_version=MODEL_VERSION,
        uptime_seconds=round(uptime, 1),
        device=_detect_device(),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-style plain-text metrics from the pipeline observer."""
    observer: PipelineObserver = app.state.observer
    summary = observer.get_summary()

    lines: list[str] = []
    lines.append("# HELP hazmat_items_total Total items classified")
    lines.append("# TYPE hazmat_items_total counter")
    lines.append(f"hazmat_items_total {summary['total_items_classified']}")

    for layer_name, layer_data in summary.get("layers", {}).items():
        prefix = f"hazmat_{layer_name}"
        lines.append(f"{prefix}_total {layer_data['total']}")
        lines.append(f"{prefix}_hazmat {layer_data['hazmat']}")
        lines.append(f"{prefix}_errors {layer_data['errors']}")
        lines.append(f"{prefix}_avg_latency_ms {layer_data['avg_latency_ms']}")

    lines.append(f"hazmat_uptime_seconds {time.time() - app.state.startup_time:.1f}")

    # Drift metrics
    drift = observer.drift_detector.get_drift_metrics()
    lines.append("")
    lines.append("# HELP hazmat_drift_psi Feature drift PSI value")
    lines.append("# TYPE hazmat_drift_psi gauge")
    lines.append(f"hazmat_drift_psi {drift.get('feature_psi', 0.0)}")
    lines.append(f"hazmat_drift_window_size {drift['window_size']}")
    lines.append(f"hazmat_drift_alerts_count {len(drift.get('alerts', {}))}")

    # Canary metrics
    canary_router: CanaryRouter = app.state.canary_router
    canary_status = canary_router.get_canary_status()
    lines.append("")
    lines.append("# HELP hazmat_canary Canary deployment metrics")
    lines.append("# TYPE hazmat_canary gauge")
    lines.append(f"hazmat_canary_active {1 if canary_status['canary_active'] else 0}")
    lines.append(f"hazmat_canary_production_total {canary_status['production']['total']}")
    lines.append(f"hazmat_canary_canary_total {canary_status['canary']['total']}")

    lines.append("")
    return "\n".join(lines)


@app.get("/drift")
async def drift_metrics() -> dict:
    """Return drift detection metrics (PSI values, alerts)."""
    observer: PipelineObserver = app.state.observer
    return observer.drift_detector.get_drift_metrics()


@app.get("/canary/status")
async def canary_status() -> dict:
    """Return canary routing status and comparison metrics."""
    canary_router: CanaryRouter = app.state.canary_router
    return canary_router.get_canary_status()


@app.get("/sample")
async def sample_items(n: int = 1) -> list[dict]:
    """Return random product samples from the dataset for testing.

    Args:
        n: Number of items to return (1-50, default 1).
    """
    n = max(1, min(n, 50))

    # Lazy-load dataset
    if not hasattr(app.state, "_sample_df") or app.state._sample_df is None:
        data_path = Path("data/processed/items_raw.parquet")
        if not data_path.exists():
            raise HTTPException(
                status_code=404, detail="Dataset not found. Run collect_data.py first."
            )
        app.state._sample_df = pd.read_parquet(data_path)
        logger.info("Loaded %d items for sampling", len(app.state._sample_df))

    df = app.state._sample_df
    sample = df.sample(n=n)
    items = []
    for _, row in sample.iterrows():
        items.append(
            {
                "title": str(row.get("title", "")),
                "description": str(row.get("description", ""))[:500],
                "category_id": str(row.get("domain_id", row.get("category_id", ""))),
            }
        )
    return items


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=PORT, reload=False)
