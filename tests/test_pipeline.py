"""Tests for HazmatPipeline (src.pipeline).

Both feature_builder and ml_classifier are fully mocked.
LLM fallback is mocked to verify it is/isn't called based on confidence.
"""

from unittest.mock import MagicMock

import numpy as np

from src.pipeline import HazmatPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(ml_confidence: float = 0.9, ml_prediction: int = 1):
    """Build a pipeline with mocked ML classifier and feature builder.

    Args:
        ml_confidence: confidence score the mock ensemble will return.
        ml_prediction: prediction value (0 or 1) the mock ensemble will return.
    """
    mock_feature_builder = MagicMock()
    mock_feature_builder._fitted = True

    def fake_transform(df, cache_path=None, update_cache=True):
        n = len(df)
        return np.random.randn(n, 100).astype(np.float32)

    mock_feature_builder.transform = MagicMock(side_effect=fake_transform)

    mock_ml = MagicMock()
    mock_ml._fitted = True

    def fake_predict_with_confidence(X):
        n = X.shape[0]
        predictions = np.full(n, ml_prediction)
        confidence = np.full(n, ml_confidence)
        return predictions, confidence

    mock_ml.predict_with_confidence = MagicMock(side_effect=fake_predict_with_confidence)

    mock_llm = MagicMock()
    mock_llm.classify = MagicMock(
        return_value={
            "is_hazmat": True,
            "confidence": 0.85,
            "reason": "LLM fallback: produto perigoso",
        }
    )

    pipeline = HazmatPipeline(
        ml_classifier=mock_ml,
        feature_builder=mock_feature_builder,
        llm_fallback=mock_llm,
        confidence_threshold=0.6,
    )
    return pipeline, mock_ml, mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifySingle:
    def test_returns_expected_keys(self):
        pipeline, _, _ = _make_pipeline(ml_confidence=0.9, ml_prediction=1)
        result = pipeline.classify_single("Gasolina 5L", "combustivel")

        assert "is_hazmat" in result
        assert "reason" in result
        assert "confidence_score" in result
        assert "source_layer" in result

    def test_returns_bool_is_hazmat(self):
        pipeline, _, _ = _make_pipeline(ml_confidence=0.9, ml_prediction=1)
        result = pipeline.classify_single("Bateria litio 18650")

        assert isinstance(result["is_hazmat"], bool)

    def test_confidence_is_float(self):
        pipeline, _, _ = _make_pipeline(ml_confidence=0.88)
        result = pipeline.classify_single("Thinner 1L")

        assert isinstance(result["confidence_score"], float)


class TestHighConfidenceNoLLM:
    def test_high_confidence_does_not_trigger_llm(self):
        """When ML confidence >= threshold, LLM should NOT be called."""
        pipeline, mock_ml, mock_llm = _make_pipeline(ml_confidence=0.9, ml_prediction=1)

        result = pipeline.classify_single("Gasolina 5L")

        assert result["source_layer"] == "ml"
        mock_llm.classify.assert_not_called()

    def test_high_confidence_returns_ml_prediction(self):
        pipeline, _, _ = _make_pipeline(ml_confidence=0.85, ml_prediction=0)

        result = pipeline.classify_single("Camiseta algodao")

        assert result["is_hazmat"] is False
        assert result["source_layer"] == "ml"


class TestLowConfidenceLLMFallback:
    def test_low_confidence_triggers_llm(self):
        """When ML confidence < threshold, LLM SHOULD be called."""
        pipeline, mock_ml, mock_llm = _make_pipeline(ml_confidence=0.3, ml_prediction=0)

        result = pipeline.classify_single("Produto quimico ambiguo")

        mock_llm.classify.assert_called_once()
        assert result["source_layer"] == "llm"

    def test_llm_result_is_used(self):
        pipeline, _, mock_llm = _make_pipeline(ml_confidence=0.2, ml_prediction=0)
        mock_llm.classify.return_value = {
            "is_hazmat": True,
            "confidence": 0.85,
            "reason": "LLM: contains dangerous chemical",
        }

        result = pipeline.classify_single("Substancia desconhecida")

        assert result["is_hazmat"] is True
        assert result["source_layer"] == "llm"


class TestNoMLModel:
    def test_no_ml_model_goes_to_llm(self):
        """When no ML model is loaded, pipeline should fall through to LLM."""
        mock_llm = MagicMock()
        mock_llm.classify.return_value = {
            "is_hazmat": False,
            "confidence": 0.80,
            "reason": "LLM: safe product",
        }

        pipeline = HazmatPipeline(
            ml_classifier=None,
            feature_builder=None,
            llm_fallback=mock_llm,
        )

        result = pipeline.classify_single("Mesa de escritorio")

        mock_llm.classify.assert_called_once()
        assert result["is_hazmat"] is False
        assert result["source_layer"] == "llm"


class TestConfidenceThreshold:
    def test_exactly_at_threshold_uses_ml(self):
        """Confidence exactly at threshold should use ML (>= comparison)."""
        pipeline, _, mock_llm = _make_pipeline(ml_confidence=0.6)
        pipeline.confidence_threshold = 0.6

        result = pipeline.classify_single("Produto teste")

        assert result["source_layer"] == "ml"
        mock_llm.classify.assert_not_called()

    def test_just_below_threshold_uses_llm(self):
        """Confidence just below threshold should trigger LLM."""
        pipeline, _, mock_llm = _make_pipeline(ml_confidence=0.59)
        pipeline.confidence_threshold = 0.6

        pipeline.classify_single("Produto teste")

        mock_llm.classify.assert_called_once()
