"""Shared fixtures for Hazmat Classifier tests.

All heavy dependencies (embedding model, ensemble, pipeline) are mocked
so tests run in seconds without downloading 1.1GB+ models.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_items() -> list[dict]:
    """Mix of hazmat and non-hazmat items for testing."""
    return [
        {
            "title": "Gasolina 5L para motor",
            "description": "Combustivel inflamavel classe 3",
            "category_id": "MLB1234",
        },
        {
            "title": "Bateria litio 18650 3.7V",
            "description": "Celula de litio recarregavel",
            "category_id": "MLB5678",
        },
        {
            "title": "Camiseta algodao masculina",
            "description": "Camiseta 100% algodao tamanho M",
            "category_id": "MLB9999",
        },
        {
            "title": "Thinner 1 litro",
            "description": "Solvente para diluicao de tintas",
            "category_id": "MLB1111",
        },
        {
            "title": "Livro Python para iniciantes",
            "description": "Aprenda Python do zero",
            "category_id": "MLB2222",
        },
    ]


@pytest.fixture
def sample_dataframe(sample_items) -> pd.DataFrame:
    """DataFrame with title, description, domain_id, item_id columns."""
    df = pd.DataFrame(sample_items)
    df = df.rename(columns={"category_id": "domain_id"})
    df["item_id"] = [f"MLB-{i}" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Mock FeatureBuilder (avoids loading 1.1GB embedding model)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_feature_builder():
    """FeatureBuilder with mocked embedding model returning random 768-dim vectors."""
    with patch("src.features.SentenceTransformer"):
        from src.features import FeatureBuilder

        builder = MagicMock(spec=FeatureBuilder)
        builder._fitted = True

        def fake_transform(df, cache_path=None):
            n_samples = len(df)
            rng = np.random.default_rng(42)
            # 768 embedding dims + 50 keyword dims + 1 category dim
            n_features = 819
            return rng.standard_normal((n_samples, n_features)).astype(np.float32)

        builder.transform = MagicMock(side_effect=fake_transform)
        builder.fit_transform = MagicMock(side_effect=fake_transform)
        yield builder


# ---------------------------------------------------------------------------
# Mock HazmatEnsemble (avoids loading XGBoost/LightGBM/RF models)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ensemble():
    """HazmatEnsemble with mocked predict methods returning fixed values."""
    from src.model import HazmatEnsemble

    ensemble = MagicMock(spec=HazmatEnsemble)
    ensemble._fitted = True
    ensemble.threshold = 0.5
    ensemble.weights = {"xgboost": 0.4, "lightgbm": 0.4, "random_forest": 0.2}

    def fake_predict_with_confidence(X):
        n = X.shape[0]
        predictions = np.array([1, 1, 0, 1, 0] * ((n // 5) + 1))[:n]
        confidence = np.array([0.92, 0.88, 0.95, 0.85, 0.90] * ((n // 5) + 1))[:n]
        return predictions, confidence

    def fake_predict_proba(X):
        n = X.shape[0]
        return np.array([0.95, 0.90, 0.05, 0.88, 0.10] * ((n // 5) + 1))[:n]

    def fake_predict_detailed(X):
        predictions, confidence = fake_predict_with_confidence(X)
        proba = fake_predict_proba(X)
        return {
            "predictions": predictions,
            "confidence": confidence,
            "ensemble_proba": proba,
            "per_model": {
                "xgboost": proba,
                "lightgbm": proba,
                "random_forest": proba,
            },
        }

    ensemble.predict_with_confidence = MagicMock(side_effect=fake_predict_with_confidence)
    ensemble.predict_proba = MagicMock(side_effect=fake_predict_proba)
    ensemble.predict_detailed = MagicMock(side_effect=fake_predict_detailed)
    yield ensemble


# ---------------------------------------------------------------------------
# Mock HazmatPipeline (avoids loading any models)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline():
    """HazmatPipeline with mocked classify_single and classify_batch."""
    from src.pipeline import HazmatPipeline

    pipeline = MagicMock(spec=HazmatPipeline)

    def fake_classify_single(title, description="", category_id=""):
        return {
            "is_hazmat": "gasolina" in title.lower() or "bateria" in title.lower(),
            "reason": "Mocked ML classification",
            "confidence_score": 0.92,
            "source_layer": "ml",
        }

    pipeline.classify_single = MagicMock(side_effect=fake_classify_single)
    pipeline.classify_batch = MagicMock(return_value=pd.DataFrame())
    yield pipeline
