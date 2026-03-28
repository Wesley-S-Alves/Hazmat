"""Tests for FastAPI endpoints (src.api).

The pipeline in app.state is fully mocked so no real models are loaded.
Uses FastAPI TestClient (synchronous, backed by httpx).
"""

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.pipeline import HazmatPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with a mocked pipeline injected into app.state.

    Bypasses the lifespan model loading by directly setting app.state attributes.
    """
    mock_pipeline = MagicMock(spec=HazmatPipeline)

    def fake_classify_single(title, description="", category_id=""):
        is_hazmat = any(kw in title.lower() for kw in ("gasolina", "bateria", "thinner"))
        return {
            "is_hazmat": is_hazmat,
            "reason": "Mocked classification",
            "confidence_score": 0.92,
            "source_layer": "ml",
        }

    def fake_classify_batch(df, use_llm=True):
        results = []
        for _, row in df.iterrows():
            is_hazmat = any(kw in row["title"].lower() for kw in ("gasolina", "bateria", "thinner"))
            results.append(
                {
                    "is_hazmat": is_hazmat,
                    "reason": "Mocked batch classification",
                    "confidence_score": 0.90,
                    "source_layer": "ml",
                    "ensemble_proba": 0.88,
                }
            )
        result_df = df.copy()
        for key in ("is_hazmat", "reason", "confidence_score", "source_layer", "ensemble_proba"):
            result_df[key] = [r[key] for r in results]
        return result_df

    mock_pipeline.classify_single = MagicMock(side_effect=fake_classify_single)
    mock_pipeline.classify_batch = MagicMock(side_effect=fake_classify_batch)
    mock_pipeline.ml_classifier = MagicMock()
    mock_pipeline.feature_builder = MagicMock()
    mock_pipeline.observer = MagicMock()
    mock_pipeline.observer.get_summary.return_value = {
        "total_items_classified": 0,
        "layers": {},
    }

    # Inject mocked pipeline and lifespan state
    app.state.pipeline = mock_pipeline
    app.state.observer = mock_pipeline.observer
    app.state.models_loaded = True
    app.state.startup_time = time.time()

    with TestClient(app, raise_server_exceptions=False) as c:
        # Re-inject mocks AFTER lifespan (lifespan may overwrite app.state)
        app.state.pipeline = mock_pipeline
        app.state.observer = mock_pipeline.observer
        app.state.models_loaded = True
        app.state.startup_time = time.time()
        yield c


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------


class TestClassifyEndpoint:
    def test_classify_returns_200(self, client):
        response = client.post(
            "/classify",
            json={
                "title": "Gasolina 5L",
                "description": "Combustivel inflamavel",
                "category_id": "MLB1234",
            },
        )
        assert response.status_code == 200

    def test_classify_response_schema(self, client):
        response = client.post(
            "/classify",
            json={
                "title": "Gasolina 5L",
            },
        )
        data = response.json()

        assert "is_hazmat" in data
        assert "reason" in data
        assert "confidence_score" in data
        assert "source_layer" in data
        assert isinstance(data["is_hazmat"], bool)
        assert isinstance(data["confidence_score"], (int, float))

    def test_classify_hazmat_item(self, client):
        response = client.post(
            "/classify",
            json={
                "title": "Gasolina 5 litros",
            },
        )
        data = response.json()
        assert data["is_hazmat"] is True

    def test_classify_safe_item(self, client):
        response = client.post(
            "/classify",
            json={
                "title": "Camiseta algodao masculina",
            },
        )
        data = response.json()
        assert data["is_hazmat"] is False

    def test_classify_missing_title_returns_422(self, client):
        """Missing required 'title' field should return 422 Unprocessable Entity."""
        response = client.post(
            "/classify",
            json={
                "description": "some description",
            },
        )
        assert response.status_code == 422

    def test_classify_empty_title_returns_422(self, client):
        """Empty string for title should fail min_length=1 validation."""
        response = client.post(
            "/classify",
            json={
                "title": "",
            },
        )
        assert response.status_code == 422

    def test_classify_optional_fields(self, client):
        """description and category_id are optional."""
        response = client.post(
            "/classify",
            json={
                "title": "Thinner 1L",
            },
        )
        assert response.status_code == 200
        assert response.json()["is_hazmat"] is True


# ---------------------------------------------------------------------------
# POST /classify/batch
# ---------------------------------------------------------------------------


class TestBatchEndpoint:
    def test_batch_returns_200(self, client):
        response = client.post(
            "/classify/batch",
            json={
                "items": [
                    {"title": "Gasolina 5L"},
                    {"title": "Camiseta branca"},
                ],
            },
        )
        assert response.status_code == 200

    def test_batch_response_has_results_and_summary(self, client):
        response = client.post(
            "/classify/batch",
            json={
                "items": [
                    {"title": "Gasolina 5L"},
                    {"title": "Camiseta branca"},
                    {"title": "Bateria litio 18650"},
                ],
            },
        )
        data = response.json()

        assert "results" in data
        assert "summary" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 3

    def test_batch_result_items_have_expected_keys(self, client):
        response = client.post(
            "/classify/batch",
            json={
                "items": [
                    {"title": "Thinner 1L"},
                ],
            },
        )
        data = response.json()
        item = data["results"][0]

        assert "is_hazmat" in item
        assert "confidence_score" in item
        assert "source_layer" in item
        assert "reason" in item

    def test_batch_summary_has_counts(self, client):
        response = client.post(
            "/classify/batch",
            json={
                "items": [
                    {"title": "Gasolina 5L"},
                    {"title": "Livro de Python"},
                ],
            },
        )
        data = response.json()
        summary = data["summary"]

        assert "total" in summary
        assert "hazmat" in summary
        assert summary["total"] == 2

    def test_batch_empty_items_returns_422(self, client):
        """BatchClassifyRequest requires min_length=1, so empty list is invalid."""
        response = client.post(
            "/classify/batch",
            json={
                "items": [],
            },
        )
        assert response.status_code == 422

    def test_batch_mixed_hazmat_results(self, client):
        response = client.post(
            "/classify/batch",
            json={
                "items": [
                    {"title": "Gasolina 5L"},
                    {"title": "Livro de Python"},
                ],
            },
        )
        data = response.json()
        results = data["results"]

        hazmat_flags = [item["is_hazmat"] for item in results]
        assert hazmat_flags[0] is True
        assert hazmat_flags[1] is False


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "models_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_health_status_ok_when_models_loaded(self, client):
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_health_status_degraded_when_no_models(self):
        """When models_loaded is False, status should be 'degraded'."""
        mock_pipeline = MagicMock(spec=HazmatPipeline)
        mock_pipeline.ml_classifier = None
        mock_pipeline.feature_builder = None
        mock_pipeline.observer = MagicMock()

        with TestClient(app) as c:
            # Override AFTER lifespan
            app.state.pipeline = mock_pipeline
            app.state.observer = mock_pipeline.observer
            app.state.models_loaded = False
            app.state.startup_time = time.time()

            response = c.get("/health")

        data = response.json()
        assert data["status"] == "degraded"
        assert data["models_loaded"] is False
