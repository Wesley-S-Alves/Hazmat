.PHONY: install test lint lint-fix serve api dashboard docker-build docker-run train classify-all mlflow validate promote models-list

install:
	uv sync --frozen

test:
	OMP_NUM_THREADS=1 uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

serve:
	@echo "Starting API (8080) + MLflow (5000) + Dashboard (7860)..."
	@OMP_NUM_THREADS=1 uv run uvicorn src.api:app --port 8080 &
	@uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 &
	@sleep 3
	@uv run python -m src.dashboard

api:
	OMP_NUM_THREADS=1 uv run uvicorn src.api:app --reload --port 8080

dashboard:
	uv run python -m src.dashboard

docker-build:
	docker build -t hazmat-classifier .

docker-run:
	docker compose up

# ── ML Engineering ──────────────────────────────────
train:
	OMP_NUM_THREADS=1 uv run python scripts/train_model.py

classify-all:
	OMP_NUM_THREADS=1 uv run python scripts/classify_all.py

validate:
	OMP_NUM_THREADS=1 uv run python scripts/validate_model.py

promote-staging:
	OMP_NUM_THREADS=1 uv run python scripts/promote_model.py --to staging

promote-production:
	OMP_NUM_THREADS=1 uv run python scripts/promote_model.py --to production

models-list:
	OMP_NUM_THREADS=1 uv run python scripts/promote_model.py --list

register-prompts:
	OMP_NUM_THREADS=1 uv run python scripts/register_prompts.py -m "$(or $(m),Update prompts)"

prompts-list:
	OMP_NUM_THREADS=1 uv run python scripts/register_prompts.py --list

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
