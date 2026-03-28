"""Observability module for the Hazmat classification pipeline.

Tracks metrics per layer, confidence distributions, latencies, and error rates.
Prometheus-compatible metrics export. Drift detection for production monitoring.
Structured logging (text or JSON format).
"""

import json
import logging
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("hazmat.observability")

# Training baseline (from 10k labels)
TRAINING_HAZMAT_RATE = 0.38
DRIFT_THRESHOLD = 0.05  # Alert if hazmat rate deviates >5%
CONFIDENCE_ALERT_THRESHOLD = 0.7  # Alert if avg confidence drops below this
PSI_ALERT_THRESHOLD = 0.2  # PSI > 0.2 = significant drift
BASELINE_PATH = Path(os.environ.get("MODELS_DIR", "models")) / "drift_baseline.json"


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1  — no significant shift
    PSI 0.1-0.2 — moderate shift
    PSI > 0.2  — significant shift (alert!)

    Uses histogram binning with a small epsilon to avoid log(0).
    """
    eps = 1e-4
    # Use the same bin edges for both distributions
    combined = np.concatenate([expected, actual])
    _, bin_edges = np.histogram(combined, bins=n_bins)

    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # Normalize to proportions
    expected_pct = expected_hist / expected_hist.sum() + eps
    actual_pct = actual_hist / actual_hist.sum() + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


@dataclass
class LayerMetrics:
    """Metrics for a single classification layer."""

    total: int = 0
    hazmat: int = 0
    not_hazmat: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    confidence_scores: list = field(default_factory=list)

    @property
    def hazmat_rate(self) -> float:
        return self.hazmat / self.total if self.total > 0 else 0.0

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

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "hazmat": self.hazmat,
            "not_hazmat": self.not_hazmat,
            "errors": self.errors,
            "hazmat_rate": round(self.hazmat_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_confidence": round(self.avg_confidence, 4),
        }


class DriftDetector:
    """Detect data/model drift via rolling statistics and PSI.

    Tracks three kinds of drift:
    1. **Prediction drift** — has the hazmat-rate shifted?
    2. **Confidence drift** — is the model less certain than during training?
    3. **Feature drift** — have embedding-space statistics shifted? (PSI)

    Baseline statistics are saved during training via ``save_baseline()``.
    At serving time the detector compares a rolling window against that baseline
    using the Population Stability Index (PSI).
    """

    def __init__(
        self,
        window_size: int = 1000,
        baseline_hazmat_rate: float = TRAINING_HAZMAT_RATE,
        drift_threshold: float = DRIFT_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_ALERT_THRESHOLD,
        psi_threshold: float = PSI_ALERT_THRESHOLD,
        baseline_path: Path | None = None,
    ):
        self._window: deque = deque(maxlen=window_size)
        self._confidence_window: deque = deque(maxlen=window_size)
        # Store raw feature vectors (only a few summary dims to save memory)
        self._feature_window: deque = deque(maxlen=window_size)
        self.baseline_hazmat_rate = baseline_hazmat_rate
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold
        self.psi_threshold = psi_threshold
        self._last_alert_time = 0.0

        # Baseline from training data (loaded from disk)
        self._baseline: dict | None = None
        self._baseline_path = baseline_path or BASELINE_PATH
        self._load_baseline()

    # ── Baseline persistence ────────────────────────────────────────────

    def _load_baseline(self) -> None:
        """Load baseline stats saved during training."""
        if self._baseline_path.exists():
            try:
                with open(self._baseline_path) as f:
                    self._baseline = json.load(f)
                # Override training hazmat rate from baseline
                self.baseline_hazmat_rate = self._baseline.get(
                    "prediction_rate", self.baseline_hazmat_rate
                )
                logger.info(
                    "Drift baseline loaded from %s (hazmat_rate=%.4f, n_samples=%d)",
                    self._baseline_path,
                    self.baseline_hazmat_rate,
                    self._baseline.get("n_samples", 0),
                )
            except Exception as exc:
                logger.warning("Could not load drift baseline: %s", exc)

    @classmethod
    def save_baseline(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        output_path: Path | None = None,
    ) -> Path:
        """Save baseline feature distribution statistics from training data.

        Called once after training so the serving-time detector has a
        reference distribution to compare against.

        Saves:
          - prediction_rate (hazmat ratio)
          - confidence distribution stats (placeholder — filled at serve time)
          - feature_mean / feature_std per embedding dimension
          - feature_sample: random sample of confidence-proxy values for PSI
        """
        output_path = output_path or BASELINE_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Feature statistics (mean/std per dimension)
        feat_mean = X.mean(axis=0).tolist()
        feat_std = X.std(axis=0).tolist()

        # Prediction distribution baseline
        hazmat_rate = float(y.mean())

        # Sample of per-row L2 norms (proxy for feature-space distribution PSI)
        norms = np.linalg.norm(X, axis=1)
        norm_sample = norms.tolist()

        # Confidence proxy: distance of ensemble proba from 0.5
        # We store norms as a stand-in; real confidence comes at serve time
        baseline = {
            "n_samples": int(len(y)),
            "prediction_rate": hazmat_rate,
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "feature_norm_sample": norm_sample,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(output_path, "w") as f:
            json.dump(baseline, f, indent=2)

        logger.info(
            "Drift baseline saved to %s (n=%d, hazmat_rate=%.4f, dims=%d)",
            output_path,
            len(y),
            hazmat_rate,
            X.shape[1],
        )
        return output_path

    # ── Recording ───────────────────────────────────────────────────────

    def record(
        self, is_hazmat: bool, confidence: float, features: np.ndarray | None = None
    ) -> None:
        """Record a single prediction for drift tracking.

        Args:
            is_hazmat: Predicted label.
            confidence: Model confidence score.
            features: Optional feature vector for feature-drift PSI.
        """
        self._window.append(1 if is_hazmat else 0)
        self._confidence_window.append(confidence)
        if features is not None:
            self._feature_window.append(float(np.linalg.norm(features)))

    # ── Drift checks ────────────────────────────────────────────────────

    def check(self) -> dict:
        """Check for drift. Returns dict with alerts (empty = no drift)."""
        alerts = {}

        if len(self._window) < 100:
            return alerts  # Not enough data

        # 1. Prediction distribution drift
        current_rate = sum(self._window) / len(self._window)
        deviation = abs(current_rate - self.baseline_hazmat_rate)
        if deviation > self.drift_threshold:
            alerts["hazmat_rate_drift"] = {
                "current": round(current_rate, 4),
                "baseline": self.baseline_hazmat_rate,
                "deviation": round(deviation, 4),
                "message": f"Hazmat rate drifted {deviation:.1%} from baseline",
            }

        # 2. Confidence drift
        avg_confidence = sum(self._confidence_window) / len(self._confidence_window)
        if avg_confidence < self.confidence_threshold:
            alerts["low_confidence"] = {
                "current_avg": round(avg_confidence, 4),
                "threshold": self.confidence_threshold,
                "message": (
                    f"Average confidence {avg_confidence:.3f} below "
                    f"threshold {self.confidence_threshold}"
                ),
            }

        # 3. Feature drift via PSI (requires baseline + enough feature samples)
        if self._baseline and len(self._feature_window) >= 100:
            baseline_norms = self._baseline.get("feature_norm_sample")
            if baseline_norms:
                psi = _compute_psi(
                    np.array(baseline_norms),
                    np.array(list(self._feature_window)),
                )
                if psi > self.psi_threshold:
                    alerts["feature_drift_psi"] = {
                        "psi": round(psi, 4),
                        "threshold": self.psi_threshold,
                        "message": (
                            f"Feature distribution PSI={psi:.4f} exceeds "
                            f"threshold {self.psi_threshold}"
                        ),
                    }

        # 4. Confidence distribution PSI (compare vs flat baseline of ~0.85 avg)
        if self._baseline and len(self._confidence_window) >= 100:
            current_confs = np.array(list(self._confidence_window))
            # Build a synthetic baseline confidence distribution centered on 0.85
            baseline_confs = np.random.RandomState(42).normal(
                loc=0.85, scale=0.1, size=max(len(current_confs), 500)
            )
            baseline_confs = np.clip(baseline_confs, 0, 1)
            conf_psi = _compute_psi(baseline_confs, current_confs)
            if conf_psi > self.psi_threshold:
                alerts["confidence_drift_psi"] = {
                    "psi": round(conf_psi, 4),
                    "threshold": self.psi_threshold,
                    "message": (
                        f"Confidence distribution PSI={conf_psi:.4f} exceeds "
                        f"threshold {self.psi_threshold}"
                    ),
                }

        # Rate-limit logging (max once per 60s)
        now = time.time()
        if alerts and (now - self._last_alert_time) > 60:
            self._last_alert_time = now
            for name, alert in alerts.items():
                logger.warning("DRIFT ALERT [%s]: %s", name, alert["message"])

        return alerts

    # ── Metrics export ──────────────────────────────────────────────────

    def get_drift_metrics(self) -> dict:
        """Return comprehensive drift metrics for the /drift endpoint."""
        n = len(self._window)
        metrics = {
            "window_size": n,
            "current_hazmat_rate": round(sum(self._window) / n, 4) if n > 0 else 0,
            "current_avg_confidence": (round(sum(self._confidence_window) / n, 4) if n > 0 else 0),
            "baseline_hazmat_rate": self.baseline_hazmat_rate,
            "baseline_loaded": self._baseline is not None,
            "baseline_n_samples": (self._baseline.get("n_samples", 0) if self._baseline else 0),
            "feature_samples_collected": len(self._feature_window),
            "alerts": self.check(),
        }

        # Include PSI values even when below threshold
        if self._baseline and len(self._feature_window) >= 100:
            baseline_norms = self._baseline.get("feature_norm_sample")
            if baseline_norms:
                metrics["feature_psi"] = round(
                    _compute_psi(
                        np.array(baseline_norms),
                        np.array(list(self._feature_window)),
                    ),
                    4,
                )

        return metrics

    def to_dict(self) -> dict:
        n = len(self._window)
        return {
            "window_size": n,
            "current_hazmat_rate": round(sum(self._window) / n, 4) if n > 0 else 0,
            "current_avg_confidence": round(sum(self._confidence_window) / n, 4) if n > 0 else 0,
            "baseline_hazmat_rate": self.baseline_hazmat_rate,
        }


class PipelineObserver:
    """Collects and reports metrics for the hazmat classification pipeline."""

    def __init__(self, output_dir: Path | None = None):
        self.layers: dict[str, LayerMetrics] = defaultdict(LayerMetrics)
        self.category_distribution: Counter = Counter()
        self.output_dir = output_dir or Path("data/output")
        self._start_time = time.time()
        self.drift_detector = DriftDetector()

        # Prometheus-style counters
        self._total_classifications = 0
        self._total_errors = 0
        self._latency_buckets = defaultdict(int)  # bucket_ms -> count

    def record_classification(
        self,
        layer: str,
        is_hazmat: bool,
        confidence: float,
        latency_ms: float,
        category_id: str | None = None,
        features: np.ndarray | None = None,
    ) -> None:
        """Record a single classification result."""
        metrics = self.layers[layer]
        metrics.total += 1
        if is_hazmat:
            metrics.hazmat += 1
        else:
            metrics.not_hazmat += 1
        metrics.total_latency_ms += latency_ms
        metrics.confidence_scores.append(confidence)

        if category_id:
            self.category_distribution[category_id] += 1

        # Update prometheus counters
        self._total_classifications += 1

        # Latency histogram buckets (ms)
        for bucket in [1, 5, 10, 50, 100, 500, 1000, 5000]:
            if latency_ms <= bucket:
                self._latency_buckets[bucket] += 1
                break

        # Drift detection (with optional feature vector for PSI)
        self.drift_detector.record(is_hazmat, confidence, features=features)

    def record_error(self, layer: str) -> None:
        """Record an error in a layer."""
        self.layers[layer].errors += 1
        self._total_errors += 1

    def get_summary(self) -> dict:
        """Get full pipeline summary."""
        total_items = sum(m.total for m in self.layers.values())
        elapsed = time.time() - self._start_time

        summary = {
            "total_items_classified": total_items,
            "elapsed_seconds": round(elapsed, 1),
            "items_per_second": round(total_items / elapsed, 2) if elapsed > 0 else 0,
            "layers": {name: m.to_dict() for name, m in self.layers.items()},
            "layer_routing": {
                name: round(m.total / total_items, 4) if total_items > 0 else 0
                for name, m in self.layers.items()
            },
            "top_categories": self.category_distribution.most_common(20),
            "drift": self.drift_detector.to_dict(),
        }
        return summary

    def log_summary(self) -> None:
        """Log the summary to structured logger."""
        summary = self.get_summary()
        logger.info("Pipeline Summary: %s", json.dumps(summary, indent=2))

    def save_metrics(self, filename: str = "pipeline_metrics.json") -> Path:
        """Save metrics to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        logger.info("Metrics saved to %s", path)
        return path

    def prometheus_metrics(self) -> str:
        """Render metrics in Prometheus exposition format."""
        lines = []
        lines.append("# HELP hazmat_classifications_total Total classifications")
        lines.append("# TYPE hazmat_classifications_total counter")
        for layer_name, m in self.layers.items():
            lines.append(
                f'hazmat_classifications_total{{layer="{layer_name}",is_hazmat="true"}} {m.hazmat}'
            )
            lines.append(
                f'hazmat_classifications_total{{layer="{layer_name}",is_hazmat="false"}} {m.not_hazmat}'
            )

        lines.append("")
        lines.append("# HELP hazmat_classification_errors_total Total errors")
        lines.append("# TYPE hazmat_classification_errors_total counter")
        lines.append(f"hazmat_classification_errors_total {self._total_errors}")

        lines.append("")
        lines.append("# HELP hazmat_classification_latency_ms Average latency in milliseconds")
        lines.append("# TYPE hazmat_classification_latency_ms gauge")
        for layer_name, m in self.layers.items():
            lines.append(
                f'hazmat_classification_latency_ms{{layer="{layer_name}"}} {m.avg_latency_ms:.2f}'
            )

        lines.append("")
        lines.append("# HELP hazmat_confidence_avg Average model confidence")
        lines.append("# TYPE hazmat_confidence_avg gauge")
        for layer_name, m in self.layers.items():
            lines.append(f'hazmat_confidence_avg{{layer="{layer_name}"}} {m.avg_confidence:.4f}')

        lines.append("")
        lines.append("# HELP hazmat_hazmat_rate Current hazmat rate")
        lines.append("# TYPE hazmat_hazmat_rate gauge")
        drift = self.drift_detector.to_dict()
        lines.append(f'hazmat_hazmat_rate{{window="rolling"}} {drift["current_hazmat_rate"]}')
        lines.append(f'hazmat_hazmat_rate{{window="baseline"}} {drift["baseline_hazmat_rate"]}')

        lines.append("")
        lines.append("# HELP hazmat_uptime_seconds Seconds since pipeline started")
        lines.append("# TYPE hazmat_uptime_seconds gauge")
        lines.append(f"hazmat_uptime_seconds {time.time() - self._start_time:.1f}")

        return "\n".join(lines) + "\n"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the pipeline.

    Set LOG_FORMAT=json for JSON output (Docker/production).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    log_format = os.environ.get("LOG_FORMAT", "text")

    if log_format == "json":
        formatter = logging.Formatter(
            fmt='{"time":"%(asctime)s","logger":"%(name)s","level":"%(levelname)s","message":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (only in non-Docker environments)
    if log_format != "json":
        try:
            from src.compat import get_project_root

            log_dir = get_project_root() / "data" / "output"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "pipeline.log")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception:
            pass  # Skip file handler if path not available (Docker, tests)

    # Silence noisy third-party loggers
    for noisy in ("google_genai", "httpx", "httpcore", "urllib3", "google.auth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
