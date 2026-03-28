"""Tests for HazmatEnsemble (src.model).

All sklearn/xgboost/lightgbm models are mocked with fakes that return
fixed probabilities, so no real training or 1GB+ model loading occurs.
"""

from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: fake models that quack like sklearn estimators
# ---------------------------------------------------------------------------


class FakeEstimator:
    """Minimal estimator that returns fixed probabilities."""

    def __init__(self, hazmat_proba: float = 0.8):
        self._hazmat_proba = hazmat_proba

    def predict(self, X):
        return (np.full(X.shape[0], self._hazmat_proba) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = np.full(X.shape[0], self._hazmat_proba)
        return np.column_stack([1 - p, p])

    def fit(self, X, y, **kwargs):
        return self


def _build_ensemble_with_fakes(hazmat_proba: float = 0.8):
    """Build a HazmatEnsemble with fake estimators (no real training)."""
    with patch("src.model.HazmatEnsemble._load_config", return_value={}):
        from src.model import HazmatEnsemble

        ensemble = HazmatEnsemble.__new__(HazmatEnsemble)

    # Set attributes that __init__ would set
    fake_xgb = FakeEstimator(hazmat_proba)
    fake_lgb = FakeEstimator(hazmat_proba)
    fake_rf = FakeEstimator(hazmat_proba)

    ensemble.xgboost = fake_xgb
    ensemble.lightgbm = fake_lgb
    ensemble.random_forest = fake_rf
    ensemble.models = {
        "xgboost": fake_xgb,
        "lightgbm": fake_lgb,
        "random_forest": fake_rf,
    }
    ensemble.weights = {"xgboost": 0.4, "lightgbm": 0.4, "random_forest": 0.2}
    ensemble.threshold = 0.5
    ensemble._fitted = True
    ensemble._calibrator = None
    ensemble._shap_explainer = None
    ensemble._feature_names = None
    return ensemble


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnsembleInit:
    def test_initialization_with_default_config(self):
        """HazmatEnsemble initializes with default params when no config file exists."""
        with patch("src.model.HazmatEnsemble._load_config", return_value={}):
            from src.model import HazmatEnsemble

            ensemble = HazmatEnsemble()

        assert ensemble.threshold == 0.5
        assert "xgboost" in ensemble.weights
        assert "lightgbm" in ensemble.weights
        assert "random_forest" in ensemble.weights
        assert ensemble._fitted is False


class TestPredictWithConfidence:
    def test_returns_correct_shapes(self):
        """predict_with_confidence returns (n_samples,) for both arrays."""
        ensemble = _build_ensemble_with_fakes(0.9)
        X = np.random.randn(10, 100).astype(np.float32)

        predictions, confidence = ensemble.predict_with_confidence(X)

        assert predictions.shape == (10,)
        assert confidence.shape == (10,)

    def test_predictions_are_binary(self):
        ensemble = _build_ensemble_with_fakes(0.9)
        X = np.random.randn(5, 50).astype(np.float32)

        predictions, _ = ensemble.predict_with_confidence(X)

        assert set(np.unique(predictions)).issubset({0, 1})

    def test_confidence_between_0_and_1(self):
        ensemble = _build_ensemble_with_fakes(0.85)
        X = np.random.randn(20, 50).astype(np.float32)

        _, confidence = ensemble.predict_with_confidence(X)

        assert np.all(confidence >= 0.0)
        assert np.all(confidence <= 1.0)

    def test_high_proba_gives_high_confidence(self):
        """When all models agree on high proba, confidence should be high."""
        ensemble = _build_ensemble_with_fakes(0.95)
        X = np.random.randn(5, 50).astype(np.float32)

        _, confidence = ensemble.predict_with_confidence(X)

        assert np.all(confidence > 0.5)

    def test_mid_proba_gives_lower_confidence(self):
        """When proba is near 0.5, confidence should be lower."""
        ensemble = _build_ensemble_with_fakes(0.52)
        X = np.random.randn(5, 50).astype(np.float32)

        _, confidence = ensemble.predict_with_confidence(X)

        # Confidence should be noticeably lower than for 0.95 proba
        assert np.all(confidence < 0.8)


class TestPredictDetailed:
    def test_returns_expected_keys(self):
        ensemble = _build_ensemble_with_fakes(0.8)
        X = np.random.randn(5, 50).astype(np.float32)

        result = ensemble.predict_detailed(X)

        assert "predictions" in result
        assert "confidence" in result
        assert "ensemble_proba" in result
        assert "per_model" in result
        assert "xgboost" in result["per_model"]
        assert "lightgbm" in result["per_model"]
        assert "random_forest" in result["per_model"]

    def test_per_model_shapes_match(self):
        ensemble = _build_ensemble_with_fakes(0.75)
        X = np.random.randn(8, 50).astype(np.float32)

        result = ensemble.predict_detailed(X)

        assert result["predictions"].shape == (8,)
        assert result["ensemble_proba"].shape == (8,)
        for name, proba in result["per_model"].items():
            assert proba.shape == (8,), f"{name} shape mismatch"


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve weights and threshold."""
        ensemble = _build_ensemble_with_fakes(0.8)
        ensemble.threshold = 0.42
        ensemble.weights = {"xgboost": 0.5, "lightgbm": 0.3, "random_forest": 0.2}

        ensemble.save(tmp_path)

        # Verify files were created
        assert (tmp_path / "xgboost.joblib").exists()
        assert (tmp_path / "lightgbm.joblib").exists()
        assert (tmp_path / "random_forest.joblib").exists()
        assert (tmp_path / "ensemble_weights.joblib").exists()
        assert (tmp_path / "ensemble_threshold.joblib").exists()

        # Load into a fresh ensemble
        loaded = _build_ensemble_with_fakes(0.5)
        loaded.load(tmp_path)

        assert loaded.threshold == 0.42
        assert loaded.weights == {"xgboost": 0.5, "lightgbm": 0.3, "random_forest": 0.2}
        assert loaded._fitted is True

    def test_load_restores_model_predictions(self, tmp_path):
        """Loaded models should produce the same predictions as saved ones."""
        ensemble = _build_ensemble_with_fakes(0.8)
        X = np.random.randn(3, 50).astype(np.float32)

        original_proba = ensemble.predict_proba(X)
        ensemble.save(tmp_path)

        loaded = _build_ensemble_with_fakes(0.1)  # different default
        loaded.load(tmp_path)
        loaded_proba = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(original_proba, loaded_proba)


class TestNotFitted:
    def test_predict_proba_raises_when_not_fitted(self):
        with patch("src.model.HazmatEnsemble._load_config", return_value={}):
            from src.model import HazmatEnsemble

            ensemble = HazmatEnsemble()

        X = np.random.randn(3, 50).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict_proba(X)
