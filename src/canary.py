"""Canary deployment router for the Hazmat Classifier.

Routes a configurable percentage of traffic to a "canary" (new) model version
while the rest goes to the "production" (current) model. Tracks metrics for
both versions independently and auto-promotes or rolls back the canary based
on observed performance.

Usage:
    router = CanaryRouter(production_dir="models", canary_dir="models_canary")
    result = router.classify(title="Lithium battery pack", description="...", category_id="123")
"""

from src.compat import configure_omp

configure_omp()

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline import HazmatPipeline

logger = logging.getLogger("hazmat.canary")


@dataclass
class VersionMetrics:
    """Track per-version classification metrics."""

    total: int = 0
    hazmat: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    confidence_scores: list = field(default_factory=list)
    predictions: list = field(default_factory=list)  # 1/0 for F1 comparison

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        return (
            sum(self.confidence_scores) / len(self.confidence_scores)
            if self.confidence_scores
            else 0.0
        )

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total > 0 else 0.0

    @property
    def hazmat_rate(self) -> float:
        return self.hazmat / self.total if self.total > 0 else 0.0

    def record(self, is_hazmat: bool, confidence: float, latency_ms: float) -> None:
        self.total += 1
        if is_hazmat:
            self.hazmat += 1
        self.confidence_scores.append(confidence)
        self.predictions.append(1 if is_hazmat else 0)
        self.total_latency_ms += latency_ms

    def record_error(self) -> None:
        self.errors += 1
        self.total += 1

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "hazmat": self.hazmat,
            "hazmat_rate": round(self.hazmat_rate, 4),
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_confidence": round(self.avg_confidence, 4),
        }


class CanaryRouter:
    """Route traffic between production and canary model versions.

    Loads two HazmatPipeline instances (production + canary) and routes a
    configurable fraction of requests to the canary.  After ``eval_after``
    canary predictions, the router compares metrics and auto-decides:

    - **Promote** if canary avg confidence >= production (within tolerance)
      and canary error rate is not higher.
    - **Rollback** if canary has a higher error rate.

    Args:
        production_dir: Path to production model artifacts.
        canary_dir:     Path to canary model artifacts.
        canary_pct:     Fraction of traffic routed to canary (0.0-1.0).
        eval_after:     Number of canary predictions before auto-evaluation.
        tolerance:      Allowed confidence gap (canary can be this much worse).
    """

    def __init__(
        self,
        production_dir: str | Path = "models",
        canary_dir: str | Path | None = None,
        canary_pct: float = 0.10,
        eval_after: int = 200,
        tolerance: float = 0.02,
    ):
        self.production_dir = Path(production_dir)
        self.canary_dir = Path(canary_dir) if canary_dir else None
        self.canary_pct = canary_pct
        self.eval_after = eval_after
        self.tolerance = tolerance

        # Pipelines
        self._production: HazmatPipeline | None = None
        self._canary: HazmatPipeline | None = None

        # Metrics
        self.production_metrics = VersionMetrics()
        self.canary_metrics = VersionMetrics()

        # State
        self._canary_active = False
        self._decision: str | None = None  # "promoted" | "rolled_back" | None
        self._decision_reason: str = ""
        self._created_at = time.time()

    # ── Loading ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load production and (optionally) canary pipelines."""
        # Production — always required
        self._production = HazmatPipeline()
        self._production.load_models(self.production_dir)
        logger.info("Production model loaded from %s", self.production_dir)

        # Canary — optional
        if self.canary_dir and self.canary_dir.exists():
            try:
                self._canary = HazmatPipeline()
                self._canary.load_models(self.canary_dir)
                self._canary_active = True
                logger.info(
                    "Canary model loaded from %s (%.0f%% traffic)",
                    self.canary_dir,
                    self.canary_pct * 100,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load canary model from %s: %s — canary disabled.",
                    self.canary_dir,
                    exc,
                )
                self._canary = None
                self._canary_active = False
        else:
            logger.info("No canary directory configured — production only.")

    # ── Classification ──────────────────────────────────────────────────

    def classify(
        self,
        title: str,
        description: str = "",
        category_id: str = "",
    ) -> dict:
        """Classify a single item, routing to production or canary.

        Returns the standard classify dict with an extra ``model_version``
        field indicating which pipeline served the request.
        """
        use_canary = (
            self._canary_active
            and self._canary is not None
            and self._decision is None  # stop routing once decided
            and random.random() < self.canary_pct
        )

        pipeline = self._canary if use_canary else self._production
        version = "canary" if use_canary else "production"
        metrics = self.canary_metrics if use_canary else self.production_metrics

        start = time.time()
        try:
            result = pipeline.classify_single(
                title=title,
                description=description,
                category_id=category_id,
            )
            latency_ms = (time.time() - start) * 1000
            metrics.record(
                is_hazmat=result["is_hazmat"],
                confidence=result["confidence_score"],
                latency_ms=latency_ms,
            )
        except Exception as exc:
            metrics.record_error()
            logger.error("Error in %s pipeline: %s", version, exc)
            # Fallback to production if canary failed
            if use_canary and self._production:
                result = self._production.classify_single(
                    title=title,
                    description=description,
                    category_id=category_id,
                )
                result["model_version"] = "production (canary-fallback)"
                return result
            raise

        result["model_version"] = version

        # Auto-evaluate after enough canary predictions
        if use_canary and self._decision is None and self.canary_metrics.total >= self.eval_after:
            self._auto_evaluate()

        return result

    # ── Auto-evaluation ─────────────────────────────────────────────────

    def _auto_evaluate(self) -> None:
        """Compare canary vs production metrics and decide promote/rollback."""
        prod = self.production_metrics
        canary = self.canary_metrics

        logger.info(
            "Canary evaluation triggered (canary=%d, production=%d predictions)",
            canary.total,
            prod.total,
        )

        # Rollback condition: canary error rate is higher
        if canary.error_rate > prod.error_rate + 0.01:
            self._decision = "rolled_back"
            self._decision_reason = (
                f"Canary error rate ({canary.error_rate:.2%}) higher than "
                f"production ({prod.error_rate:.2%})"
            )
            self._canary_active = False
            logger.warning("CANARY ROLLED BACK: %s", self._decision_reason)
            return

        # Promote condition: canary confidence >= production (within tolerance)
        canary_conf = canary.avg_confidence
        prod_conf = prod.avg_confidence if prod.total > 0 else 0.0

        if canary_conf >= prod_conf - self.tolerance:
            self._decision = "promoted"
            self._decision_reason = (
                f"Canary confidence ({canary_conf:.4f}) >= production "
                f"({prod_conf:.4f}) within tolerance ({self.tolerance})"
            )
            # Swap: canary becomes production
            self._production = self._canary
            self._canary = None
            self._canary_active = False
            logger.info("CANARY PROMOTED: %s", self._decision_reason)
            return

        # Neither condition met — keep running
        logger.info(
            "Canary evaluation inconclusive (canary_conf=%.4f, prod_conf=%.4f). "
            "Continuing canary traffic.",
            canary_conf,
            prod_conf,
        )

    # ── Status ──────────────────────────────────────────────────────────

    def get_canary_status(self) -> dict:
        """Return canary routing status and comparison metrics."""
        status = {
            "canary_active": self._canary_active,
            "canary_pct": self.canary_pct,
            "eval_after": self.eval_after,
            "decision": self._decision,
            "decision_reason": self._decision_reason,
            "uptime_seconds": round(time.time() - self._created_at, 1),
            "production": self.production_metrics.to_dict(),
            "canary": self.canary_metrics.to_dict(),
        }

        # Comparison summary
        if self.production_metrics.total > 0 and self.canary_metrics.total > 0:
            status["comparison"] = {
                "confidence_diff": round(
                    self.canary_metrics.avg_confidence - self.production_metrics.avg_confidence,
                    4,
                ),
                "latency_diff_ms": round(
                    self.canary_metrics.avg_latency_ms - self.production_metrics.avg_latency_ms,
                    2,
                ),
                "hazmat_rate_diff": round(
                    self.canary_metrics.hazmat_rate - self.production_metrics.hazmat_rate,
                    4,
                ),
                "error_rate_diff": round(
                    self.canary_metrics.error_rate - self.production_metrics.error_rate,
                    4,
                ),
            }

        return status
