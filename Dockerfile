# ── Stage 1: builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/       ./src/
COPY configs/   ./configs/
COPY prompts/   ./prompts/
COPY scripts/   ./scripts/

# Models are mounted at runtime (1.1 GB+), never baked into the image
ENV PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    HAZMAT_DEVICE=cpu \
    MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
