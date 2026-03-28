"""Ensemble hazmat classification model.

XGBoost + LightGBM + Random Forest with soft voting.
Trained on hybrid labels (LLM + manual).
"""

from src.compat import configure_omp, get_project_root

configure_omp()

import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`")

import joblib
import mlflow
import numpy as np
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger("hazmat.model")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", get_project_root() / "models"))


class HazmatEnsemble:
    """Ensemble classifier: XGBoost + LightGBM + Random Forest with soft voting."""

    DEFAULT_CONFIG = Path("data/output/best_ensemble_config.json")

    def __init__(self, config_path: Path | str | None = None):
        config = self._load_config(config_path)

        xgb_params = config.get("xgboost", {})
        lgb_params = config.get("lightgbm", {})
        rf_params = config.get("random_forest", {})

        self.xgboost = XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 300),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=42,
            nthread=1,
            n_jobs=1,
        )
        self.lightgbm = LGBMClassifier(
            n_estimators=lgb_params.get("n_estimators", 300),
            max_depth=lgb_params.get("max_depth", 6),
            learning_rate=lgb_params.get("learning_rate", 0.1),
            subsample=lgb_params.get("subsample", 0.8),
            colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 300),
            max_depth=rf_params.get("max_depth", 15),
            min_samples_leaf=rf_params.get("min_samples_leaf", 5),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.models = {
            "xgboost": self.xgboost,
            "lightgbm": self.lightgbm,
            "random_forest": self.random_forest,
        }

        # Load weights from config or use defaults
        w = config.get("ensemble_weights", [0.4, 0.4, 0.2])
        if isinstance(w, list) and len(w) == 3:
            self.weights = {"xgboost": w[0], "lightgbm": w[1], "random_forest": w[2]}
        elif isinstance(w, dict):
            self.weights = w
        else:
            self.weights = {"xgboost": 0.4, "lightgbm": 0.4, "random_forest": 0.2}

        self.threshold = config.get("threshold", 0.5)
        self._fitted = False
        self._calibrator = None  # Platt Scaling (isotonic calibration)
        self._shap_explainer = None  # SHAP TreeExplainer for XGBoost
        self._feature_names = None  # Feature names for SHAP explanations

        if config:
            logger.info(
                "Loaded ensemble config: XGB=%s, LGB=%s, RF=%s, weights=%s, threshold=%.3f",
                xgb_params,
                lgb_params,
                rf_params,
                self.weights,
                self.threshold,
            )

    def _load_config(self, config_path: Path | str | None = None) -> dict:
        """Load hyperparameters from Optuna config file if available."""
        import json

        path = Path(config_path) if config_path else self.DEFAULT_CONFIG
        if path.exists():
            with open(path) as f:
                config = json.load(f)
            logger.info(
                "Loaded best ensemble config from %s (score=%.4f, FN_rate=%.1f%%)",
                path,
                config.get("score", 0),
                config.get("fn_rate", 0) * 100,
            )
            return config
        logger.info("No ensemble config found at %s, using defaults", path)
        return {}

    def train(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, track_mlflow: bool = True
    ) -> dict:
        """Train all models and compute ensemble metrics.

        Returns:
            Dict with per-model and ensemble metrics.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / n_pos if n_pos > 0 else 1.0

        logger.info(
            "Training set: %d items (%d hazmat, %d non-hazmat, ratio=%.2f)",
            len(y_train),
            n_pos,
            n_neg,
            scale,
        )

        # Adjust class weights
        self.xgboost.set_params(scale_pos_weight=scale)
        self.lightgbm.set_params(scale_pos_weight=scale)

        metrics = {}

        # Train each model
        for name, model in self.models.items():
            logger.info("Training %s...", name)
            if name == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif name == "lightgbm":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)

            pred = model.predict(X_val)
            f1 = f1_score(y_val, pred)
            precision = precision_score(y_val, pred)
            recall = recall_score(y_val, pred)

            metrics[name] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
            logger.info("  %s: F1=%.4f  Precision=%.4f  Recall=%.4f", name, f1, precision, recall)

        # Ensemble (soft voting with optimized threshold)
        ensemble_proba = self._ensemble_proba(X_val)
        ensemble_pred = (ensemble_proba >= self.threshold).astype(int)
        logger.info("Using decision threshold: %.3f", self.threshold)
        ensemble_f1 = f1_score(y_val, ensemble_pred)
        ensemble_precision = precision_score(y_val, ensemble_pred)
        ensemble_recall = recall_score(y_val, ensemble_pred)

        metrics["ensemble"] = {
            "f1": ensemble_f1,
            "precision": ensemble_precision,
            "recall": ensemble_recall,
        }

        logger.info("")
        logger.info(
            "Ensemble: F1=%.4f  Precision=%.4f  Recall=%.4f",
            ensemble_f1,
            ensemble_precision,
            ensemble_recall,
        )
        logger.info("")
        logger.info(
            "Ensemble Classification Report:\n%s",
            classification_report(y_val, ensemble_pred, target_names=["Non-Hazmat", "Hazmat"]),
        )
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_val, ensemble_pred))

        # False negatives analysis (worst error)
        fn_mask = (y_val == 1) & (ensemble_pred == 0)
        fn_count = fn_mask.sum()
        fn_rate = fn_count / y_val.sum() if y_val.sum() > 0 else 0
        metrics["ensemble"]["false_negatives"] = int(fn_count)
        metrics["ensemble"]["false_negative_rate"] = float(fn_rate)
        logger.info(
            "False negatives (hazmat missed): %d (%.1f%% of all hazmat)", fn_count, fn_rate * 100
        )

        # --- Platt Scaling (probability calibration via sigmoid/logistic) ---
        logger.info("Fitting Platt Scaling calibrator...")
        from sklearn.linear_model import LogisticRegression

        raw_proba = ensemble_proba.reshape(-1, 1)
        self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._calibrator.fit(raw_proba, y_val)

        cal_proba = self._calibrator.predict_proba(raw_proba)[:, 1]
        cal_pred = (cal_proba >= self.threshold).astype(int)
        cal_f1 = f1_score(y_val, cal_pred)
        logger.info("Platt Scaling: calibrated F1=%.4f (raw=%.4f)", cal_f1, ensemble_f1)
        metrics["ensemble"]["calibrated_f1"] = cal_f1

        # --- SHAP explainer (on XGBoost, fastest for tree models) ---
        logger.info("Fitting SHAP TreeExplainer on XGBoost...")
        self._shap_explainer = shap.TreeExplainer(self.xgboost)
        logger.info("SHAP explainer ready")

        # MLflow tracking
        if track_mlflow:
            try:
                mlflow.set_experiment("hazmat-classifier")
                with mlflow.start_run(run_name="ensemble-training"):
                    mlflow.log_params(
                        {
                            "n_estimators": 300,
                            "train_size": len(y_train),
                            "val_size": len(y_val),
                            "hazmat_ratio": float(n_pos / len(y_train)),
                            "ensemble_weights": str(self.weights),
                        }
                    )
                    flat_metrics = {}
                    for model_name, model_metrics in metrics.items():
                        for metric_name, value in model_metrics.items():
                            flat_metrics[f"{model_name}_{metric_name}"] = value
                    mlflow.log_metrics(flat_metrics)
            except Exception as e:
                logger.warning("MLflow tracking failed: %s", e)

        self._fitted = True
        return metrics

    def _ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted average probability across all models."""
        total_weight = sum(self.weights.values())
        proba = np.zeros(X.shape[0])
        for name, model in self.models.items():
            weight = self.weights[name] / total_weight
            proba += weight * model.predict_proba(X)[:, 1]
        return proba

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Return ensemble probability of hazmat class.

        Args:
            calibrated: If True and calibrator is available, return Platt-scaled probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        raw = self._ensemble_proba(X)
        if calibrated and self._calibrator is not None:
            return self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw

    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions and confidence scores.

        Confidence reflects how certain the model is about its prediction:
        - proba near 0 or 1 = high confidence
        - proba near 0.5 = maximum uncertainty
        - Model agreement boosts confidence, disagreement penalizes it

        Note: threshold is used for the prediction decision, but confidence
        is based on distance from 0.5 (uncertainty midpoint).
        """
        proba = self.predict_proba(X)
        predictions = (proba >= self.threshold).astype(int)

        # Base confidence: distance from 0.5 (maximum uncertainty point)
        # proba=0.0 or 1.0 → confidence=1.0, proba=0.5 → confidence=0.0
        base_confidence = np.abs(proba - 0.5) * 2
        base_confidence = np.clip(base_confidence, 0, 1)

        # Agreement: how much individual models agree with each other
        individual_probas = []
        for name, model in self.models.items():
            individual_probas.append(model.predict_proba(X)[:, 1])
        individual_probas = np.array(individual_probas)

        # Standard deviation across models (low = agreement, high = disagreement)
        proba_std = np.std(individual_probas, axis=0)
        # Max possible std for 3 models with probas in [0,1] is ~0.47
        # Normalize: std=0 → agreement=1.0, std=0.47 → agreement=0.0
        agreement_factor = np.clip(1.0 - (proba_std / 0.47), 0, 1)

        confidence = 0.6 * base_confidence + 0.4 * agreement_factor
        return predictions, confidence

    def predict_detailed(self, X: np.ndarray) -> dict:
        """Return detailed predictions with per-model breakdown."""
        ensemble_proba = self.predict_proba(X)
        predictions, confidence = self.predict_with_confidence(X)

        per_model = {}
        for name, model in self.models.items():
            per_model[name] = model.predict_proba(X)[:, 1]

        return {
            "predictions": predictions,
            "confidence": confidence,
            "ensemble_proba": ensemble_proba,
            "per_model": per_model,
        }

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for SHAP explanations."""
        self._feature_names = names

    def explain(self, X: np.ndarray, top_k: int = 3) -> list[dict]:
        """Explain predictions using SHAP values.

        Memory-efficient: processes chunks and frees SHAP values after extracting reasons.
        Returns a list of explanations (one per sample), each containing:
        - top_features: list of (feature_name, shap_value) tuples (top contributors)
        - reason: human-readable string explaining the classification

        Falls back to keyword-based reasons if SHAP is not available.
        """
        import gc

        results = []
        n_samples = X.shape[0]
        proba_all = self.predict_proba(X)

        if self._shap_explainer is None:
            for i in range(n_samples):
                reason = self._build_reason_from_keywords(X[i], proba_all[i])
                results.append({"top_features": [], "reason": reason})
            return results

        # Process in small chunks to control memory
        CHUNK = 500
        for chunk_start in range(0, n_samples, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n_samples)
            X_chunk = X[chunk_start:chunk_end]
            proba_chunk = proba_all[chunk_start:chunk_end]

            shap_values = self._shap_explainer.shap_values(X_chunk)

            for i in range(len(X_chunk)):
                sv = shap_values[i]
                reason = self._build_reason_from_shap(sv, X_chunk[i], proba_chunk[i], top_k)
                results.append({"top_features": [], "reason": reason})

            # Free memory immediately
            del shap_values
            gc.collect()

        return results

    # Mapping: keyword prefix → hazard class display name (English)
    KEYWORD_HAZARD_CLASS = {
        "inflamável": "flammable products",
        "inflamavel": "flammable products",
        "combustível": "flammable products",
        "combustivel": "flammable products",
        "gasolina": "flammable products",
        "diesel": "flammable products",
        "querosene": "flammable products",
        "acetona": "flammable products",
        "thinner": "flammable products",
        "tíner": "flammable products",
        "solvente": "flammable products",
        "aguarrás": "flammable products",
        "etanol": "flammable products",
        "álcool": "flammable products",
        "alcool": "flammable products",
        "verniz": "flammable products",
        "benzina": "flammable products",
        "nafta": "flammable products",
        "tinta": "flammable products",
        "esmalte": "flammable products",
        "bateria": "lithium/alkaline batteries",
        "battery": "lithium/alkaline batteries",
        "lítio": "lithium/alkaline batteries",
        "litio": "lithium/alkaline batteries",
        "lithium": "lithium/alkaline batteries",
        "li-ion": "lithium/alkaline batteries",
        "lipo": "lithium/alkaline batteries",
        "18650": "lithium/alkaline batteries",
        "pilha": "lithium/alkaline batteries",
        "ácido": "corrosive chemicals",
        "acido": "corrosive chemicals",
        "soda_caustica": "corrosive chemicals",
        "caustica": "corrosive chemicals",
        "cloro": "corrosive chemicals",
        "hipoclorito": "corrosive chemicals",
        "desentupidor": "corrosive chemicals",
        "alvejante": "corrosive chemicals",
        "água_sanitária": "corrosive chemicals",
        "inseticida": "toxic substances",
        "pesticida": "toxic substances",
        "herbicida": "toxic substances",
        "raticida": "toxic substances",
        "veneno": "toxic substances",
        "formicida": "toxic substances",
        "fungicida": "toxic substances",
        "larvicida": "toxic substances",
        "defensivo": "toxic substances",
        "agrotóxico": "toxic substances",
        "explosiv": "explosive materials",
        "pólvora": "explosive materials",
        "polvora": "explosive materials",
        "munição": "explosive materials",
        "municao": "explosive materials",
        "espoleta": "explosive materials",
        "fogos": "explosive materials",
        "artifício": "explosive materials",
        "gás": "compressed gases",
        "gas": "compressed gases",
        "propano": "compressed gases",
        "butano": "compressed gases",
        "extintor": "compressed gases",
        "cilindro": "compressed gases",
        "aerossol": "compressed gases",
        "spray": "compressed gases",
        "oxigênio": "compressed gases",
        "oxigenio": "compressed gases",
        "peróxido": "oxidizing agents",
        "peroxido": "oxidizing agents",
        "permanganato": "oxidizing agents",
        "oxidante": "oxidizing agents",
        "óleo_motor": "miscellaneous hazmat",
        "oleo_motor": "miscellaneous hazmat",
        "fluido_freio": "miscellaneous hazmat",
        "graxa": "miscellaneous hazmat",
        "lubrificante": "miscellaneous hazmat",
        "impermeabilizante": "miscellaneous hazmat",
        "resina": "miscellaneous hazmat",
    }

    def _detect_hazard_class(self, x: np.ndarray) -> str | None:
        """Detect hazard class from active keyword features."""
        if not self._feature_names:
            return None

        kw_start = 768
        kw_end = len(x) - 1
        active_kws = []
        for ki in range(kw_start, min(kw_end, len(x))):
            if x[ki] > 0:
                kw_name = self._feature_names[ki].replace("kw_", "")
                if not kw_name.startswith("derived_"):
                    active_kws.append(kw_name)

        if not active_kws:
            return None

        # Find the hazard class from the first matching keyword
        for kw in active_kws:
            for prefix, hazard_class in self.KEYWORD_HAZARD_CLASS.items():
                if prefix in kw.lower():
                    return hazard_class
        return None

    def _build_reason_from_shap(
        self, sv: np.ndarray, x: np.ndarray, proba: float, top_k: int = 3
    ) -> str:
        """Build human-readable reason from SHAP values + hazard class detection."""
        is_haz = proba >= self.threshold
        label = "Hazmat" if is_haz else "Non-hazmat"

        # Detect hazard class from keyword features
        hazard_class = self._detect_hazard_class(x)

        if is_haz:
            if hazard_class:
                return f"{label} (conf: {proba:.2f}) — semantic similarity to {hazard_class}"
            else:
                return f"{label} (conf: {proba:.2f}) — semantic text analysis indicates hazardous material"
        else:
            if hazard_class:
                # Edge case: non-hazmat but keywords matched (false positive territory)
                return f"{label} (conf: {proba:.2f}) — despite keyword matches, overall pattern indicates safe product"
            else:
                return f"{label} (conf: {proba:.2f}) — no hazardous material indicators detected"

    def _build_reason_from_keywords(self, x: np.ndarray, proba: float) -> str:
        """Build reason from keyword features when SHAP is unavailable."""
        is_haz = proba >= self.threshold
        label = "Hazmat" if is_haz else "Non-hazmat"
        hazard_class = self._detect_hazard_class(x)

        if is_haz:
            if hazard_class:
                return f"{label} (conf: {proba:.2f}) — semantic similarity to {hazard_class}"
            else:
                return f"{label} (conf: {proba:.2f}) — semantic text analysis indicates hazardous material"
        else:
            return f"{label} (conf: {proba:.2f}) — no hazardous material indicators detected"

    def save(self, path: Path | None = None) -> None:
        """Save all trained models to disk."""
        path = path or MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, path / f"{name}.joblib")
        joblib.dump(self.weights, path / "ensemble_weights.joblib")
        joblib.dump(self.threshold, path / "ensemble_threshold.joblib")
        if self._calibrator is not None:
            joblib.dump(self._calibrator, path / "calibrator.joblib")
        if self._shap_explainer is not None:
            joblib.dump(self._shap_explainer, path / "shap_explainer.joblib")
        if self._feature_names is not None:
            joblib.dump(self._feature_names, path / "feature_names.joblib")
        logger.info("Ensemble models saved to %s (threshold=%.3f)", path, self.threshold)

    def load(self, path: Path | None = None) -> None:
        """Load trained models from disk."""
        path = path or MODELS_DIR
        for name in self.models:
            model_path = path / f"{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info("Loaded %s from %s", name, model_path)
            else:
                logger.warning("Model file not found: %s", model_path)

        # Update references
        self.xgboost = self.models["xgboost"]
        self.lightgbm = self.models["lightgbm"]
        self.random_forest = self.models["random_forest"]

        weights_path = path / "ensemble_weights.joblib"
        if weights_path.exists():
            self.weights = joblib.load(weights_path)

        threshold_path = path / "ensemble_threshold.joblib"
        if threshold_path.exists():
            self.threshold = joblib.load(threshold_path)

        cal_path = path / "calibrator.joblib"
        if cal_path.exists():
            self._calibrator = joblib.load(cal_path)
            logger.info("Loaded Platt Scaling calibrator")

        shap_path = path / "shap_explainer.joblib"
        if shap_path.exists():
            self._shap_explainer = joblib.load(shap_path)
            logger.info("Loaded SHAP explainer")

        names_path = path / "feature_names.joblib"
        if names_path.exists():
            self._feature_names = joblib.load(names_path)

        self._fitted = True
        logger.info(
            "Ensemble loaded from %s (weights=%s, threshold=%.3f)",
            path,
            self.weights,
            self.threshold,
        )
