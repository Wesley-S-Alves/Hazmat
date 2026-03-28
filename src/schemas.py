"""Pydantic models for Hazmat Classifier API request/response validation."""

from pydantic import BaseModel, Field

# ── Requests ────────────────────────────────────────────────────────────────


class ClassifyRequest(BaseModel):
    """Single item classification request."""

    title: str = Field(..., min_length=1, description="Product title")
    description: str = Field(default="", description="Product description")
    category_id: str = Field(default="", description="Marketplace category ID")


class BatchClassifyRequest(BaseModel):
    """Batch classification request (up to 100 items)."""

    items: list[ClassifyRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of items to classify (max 100)",
    )


# ── Responses ───────────────────────────────────────────────────────────────


class ClassifyResponse(BaseModel):
    """Single item classification result."""

    is_hazmat: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_layer: str = Field(description="Which layer resolved: ml, llm, or default")
    reason: str
    ensemble_proba: float | None = Field(
        default=None,
        description="Raw ensemble probability (batch only)",
    )
    per_model: dict[str, float] | None = Field(
        default=None,
        description="Per-model probabilities (batch only)",
    )


class BatchClassifyResponse(BaseModel):
    """Batch classification result."""

    results: list[ClassifyResponse]
    summary: dict = Field(
        description="Batch summary with total, hazmat, ml_resolved, llm_fallback",
    )


class HealthResponse(BaseModel):
    """Service health check."""

    status: str
    models_loaded: bool
    model_version: str
    uptime_seconds: float
    device: str
