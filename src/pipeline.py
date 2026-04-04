"""Hazmat Classification Pipeline.

Ensemble ML (embeddings + keywords as features) with LLM fallback for low confidence.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.features import FeatureBuilder
from src.llm_fallback import GeminiFallback
from src.model import HazmatEnsemble
from src.observability import PipelineObserver

logger = logging.getLogger("hazmat.pipeline")

# Confidence threshold: below this → send to LLM
CONFIDENCE_THRESHOLD = 0.6


class HazmatPipeline:
    """Hazmat classification: Ensemble ML + LLM fallback for low confidence."""

    def __init__(
        self,
        ml_classifier: HazmatEnsemble | None = None,
        feature_builder: FeatureBuilder | None = None,
        llm_fallback: GeminiFallback | None = None,
        observer: PipelineObserver | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.ml_classifier = ml_classifier
        self.feature_builder = feature_builder
        self.llm_fallback = llm_fallback or GeminiFallback()
        self.observer = observer or PipelineObserver()
        self.confidence_threshold = confidence_threshold

    def load_models(self, models_dir: Path | None = None):
        """Load pre-trained ensemble and feature builder."""
        models_dir = models_dir or Path("models")
        self.ml_classifier = HazmatEnsemble()
        self.ml_classifier.load(models_dir)
        self.feature_builder = FeatureBuilder()
        self.feature_builder.load(models_dir)

    def classify_single(self, title: str, description: str = "", category_id: str = "") -> dict:
        """Classify a single item: ML first, LLM if low confidence."""
        if not self.ml_classifier or not self.feature_builder:
            # No ML model — go straight to LLM
            return self._classify_via_llm(title, description, category_id)

        # ML Ensemble
        import hashlib

        start = time.time()
        # Generate unique item_id from content to avoid cache collisions
        content_hash = hashlib.md5(f"{title}|{description}|{category_id}".encode()).hexdigest()[:12]
        row_df = pd.DataFrame(
            [
                {
                    "item_id": f"rt_{content_hash}",
                    "title": title,
                    "description": description,
                    "category_id": category_id,
                    "domain_id": category_id,
                }
            ]
        )
        X = self.feature_builder.transform(row_df, update_cache=False)
        predictions, confidence = self.ml_classifier.predict_with_confidence(X)
        ml_latency = (time.time() - start) * 1000

        pred = predictions[0]
        conf = confidence[0]

        if conf >= self.confidence_threshold:
            is_hazmat = bool(pred)

            # SHAP explanation for the reason field
            reason = f"Ensemble ML (confidence: {conf:.2f})"
            try:
                explanations = self.ml_classifier.explain(X, top_k=3)
                if explanations and explanations[0].get("reason"):
                    reason = explanations[0]["reason"]
            except Exception:
                pass  # Fallback to generic reason

            self.observer.record_classification(
                layer="ml",
                is_hazmat=is_hazmat,
                confidence=float(conf),
                latency_ms=ml_latency,
                category_id=category_id,
            )
            return {
                "is_hazmat": is_hazmat,
                "reason": reason,
                "confidence_score": float(conf),
                "source_layer": "ml",
            }

        # Low confidence → LLM fallback
        return self._classify_via_llm(title, description, category_id)

    def _classify_via_llm(self, title: str, description: str, category_id: str) -> dict:
        """Classify via LLM fallback."""
        if not self.llm_fallback:
            return {
                "is_hazmat": False,
                "reason": "No LLM fallback available - defaulted to non-hazmat",
                "confidence_score": 0.0,
                "source_layer": "default",
            }

        start = time.time()
        llm_result = self.llm_fallback.classify(title, description)
        llm_latency = (time.time() - start) * 1000

        confidence = llm_result.get("confidence", 0.85)
        self.observer.record_classification(
            layer="llm",
            is_hazmat=llm_result["is_hazmat"],
            confidence=confidence,
            latency_ms=llm_latency,
            category_id=category_id,
        )
        return {
            "is_hazmat": llm_result["is_hazmat"],
            "reason": llm_result["reason"],
            "confidence_score": confidence,
            "source_layer": "llm",
        }

    def classify_batch(self, df: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
        """Classify a batch of items: ML ensemble + LLM fallback for low confidence.

        Args:
            df: DataFrame with columns: item_id, title, description, domain_id/category_id
            use_llm: Whether to use LLM for low-confidence items

        Returns:
            DataFrame with classification results appended
        """
        logger.info("Starting batch classification of %d items...", len(df))

        if not self.ml_classifier or not self.feature_builder:
            logger.error("No ML model loaded. Run train_model.py first.")
            return df

        # Compute features for all items at once
        start = time.time()
        X = self.feature_builder.transform(df, update_cache=False)
        feat_time = time.time() - start
        logger.info("Features computed in %.1fs", feat_time)

        # Ensemble prediction
        start = time.time()
        detailed = self.ml_classifier.predict_detailed(X)
        ml_time = time.time() - start
        logger.info("Ensemble prediction in %.1fs", ml_time)

        predictions = detailed["predictions"]
        confidence = detailed["confidence"]
        ensemble_proba = detailed["ensemble_proba"]

        # Assign results
        df = df.copy()
        df["is_hazmat"] = predictions.astype(bool)
        df["confidence_score"] = confidence
        df["ensemble_proba"] = ensemble_proba
        df["source_layer"] = "ml"
        # SHAP explanations for batch (chunked with progress logging)
        logger.info("Computing SHAP explanations for %d items...", len(df))
        shap_start = time.time()
        all_reasons = []
        SHAP_CHUNK = 1000
        try:
            for i in range(0, X.shape[0], SHAP_CHUNK):
                chunk_X = X[i : i + SHAP_CHUNK]
                chunk_expl = self.ml_classifier.explain(chunk_X, top_k=3)
                all_reasons.extend([e["reason"] for e in chunk_expl])
                done = min(i + SHAP_CHUNK, X.shape[0])
                if done % 5000 == 0 or done == X.shape[0]:
                    elapsed = time.time() - shap_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (X.shape[0] - done) / rate if rate > 0 else 0
                    logger.info(
                        "  SHAP progress: %d/%d (%.1f%%) | %.0f items/s | ETA: %.0fs",
                        done,
                        X.shape[0],
                        100 * done / X.shape[0],
                        rate,
                        eta,
                    )
            df["reason"] = all_reasons
            logger.info("SHAP explanations computed in %.1fs", time.time() - shap_start)
        except Exception as e:
            logger.warning("SHAP failed (%s), using generic reasons", e)
            df["reason"] = [f"Ensemble ML (confidence: {c:.2f})" for c in confidence]

        # Per-model probabilities for transparency
        for model_name, probas in detailed["per_model"].items():
            df[f"proba_{model_name}"] = probas

        # Record ML classifications
        for i in range(len(df)):
            cat_id = df.iloc[i].get("domain_id", df.iloc[i].get("category_id", ""))
            self.observer.record_classification(
                layer="ml",
                is_hazmat=bool(predictions[i]),
                confidence=float(confidence[i]),
                latency_ms=(ml_time * 1000) / len(df),
                category_id=str(cat_id),
            )

        # Identify low-confidence items for LLM
        low_conf_mask = confidence < self.confidence_threshold
        n_low = low_conf_mask.sum()
        logger.info(
            "ML resolved: %d items (%.1f%% high confidence)",
            len(df) - n_low,
            100 * (1 - n_low / len(df)),
        )
        logger.info("Low confidence items: %d (%.1f%%)", n_low, 100 * n_low / len(df))

        # LLM fallback for low-confidence items
        if use_llm and n_low > 0 and self.llm_fallback:
            logger.info(
                "Sending %d low-confidence items to LLM (async, %d items/req)...",
                n_low,
                self.llm_fallback.items_per_request,
            )
            low_conf_items = df[low_conf_mask][["item_id", "title", "description"]].to_dict(
                "records"
            )

            llm_start = time.time()
            llm_results = self.llm_fallback.classify_batch(low_conf_items)
            llm_time = time.time() - llm_start

            logger.info("LLM classified %d items in %.1fs", len(llm_results), llm_time)

            # Update low-confidence items with LLM results
            llm_by_id = {r["item_id"]: r for r in llm_results}
            for idx in df[low_conf_mask].index:
                item_id = df.at[idx, "item_id"]
                if item_id in llm_by_id:
                    r = llm_by_id[item_id]
                    df.at[idx, "is_hazmat"] = r["is_hazmat"]
                    df.at[idx, "reason"] = r["reason"]
                    df.at[idx, "confidence_score"] = r.get("confidence", 0.85)
                    df.at[idx, "source_layer"] = "llm"

                    # LLM failure → force human review
                    if r.get("needs_human_review"):
                        df.at[idx, "needs_human_review"] = True

                    self.observer.record_classification(
                        layer="llm",
                        is_hazmat=r["is_hazmat"],
                        confidence=r.get("confidence", 0.85),
                        latency_ms=(llm_time * 1000) / n_low,
                        category_id=str(
                            df.at[idx, "domain_id"]
                            if "domain_id" in df.columns
                            else df.at[idx, "category_id"]
                            if "category_id" in df.columns
                            else ""
                        ),
                    )
        elif n_low > 0 and not use_llm:
            logger.info("LLM disabled. %d low-confidence items kept with ML prediction.", n_low)
            df.loc[low_conf_mask, "reason"] = df.loc[low_conf_mask, "reason"].apply(
                lambda r: r + " (low confidence, LLM skipped)"
            )

        self.observer.log_summary()
        return df
