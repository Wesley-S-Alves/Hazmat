# Hazmat Classifier

Classifies Mercado Libre products as hazardous materials (hazmat) using an ensemble ML pipeline with semantic embeddings and LLM fallback.

| Metric | Value |
|--------|-------|
| Ensemble F1 | 0.934 |
| Recall | 95.5% |
| False Negative Rate | 4.5% |
| ML Resolved | 94.3% |
| LLM Fallback | 5.7% |
| Human Review | 0.07% |
| Failed Items | 0 |

## Dataset

The full classified dataset (~100k items) is available at:

**[`data/output/hazmat_classified_100k.csv`](https://github.com/Wesley-S-Alves/Hazmat/blob/main/data/output/hazmat_classified_100k.csv)**

Contains real Mercado Libre product listings classified by the pipeline, including title, description, category, classification result, confidence score, and SHAP-based explanation.

## Architecture

```
  Item (title + description + category)
          |
          v
  +-------------------------------+
  |  Fine-tuned Embeddings        |
  |  multilingual-e5-base (768d)  |
  +-------------------------------+
          |
          + keyword features (139d, binary)
          + category encoding (1d)
          |
          v
  +-------------------------------+
  |  Ensemble ML                  |
  |  XGBoost + LightGBM + RF      |
  |  (Optuna-tuned, soft voting)  |
  +---------------+---------------+
                  |
      +-----------+-----------+
      |                       |
   HIGH conf (>= 0.8)    LOW conf (< 0.8)
      |                       |
      v                       v
   ML result            Gemini 3.0 Flash
                        (async, 20 items/req)
                              |
                              v
                        LLM result
```

## Quick Start

```bash
make install                # Install dependencies (requires uv)
cp .env.example .env        # Configure GEMINI_API_KEY
make serve                  # API :8080 | MLflow :5000 | Dashboard :7860
```

**Classify a product:**

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"title": "Bateria de litio 18650 3.7V"}'
```

## Services

| Service | Port | Command |
|---------|------|---------|
| FastAPI API | 8080 | `make api` |
| MLflow UI | 5000 | `make mlflow` |
| Gradio Dashboard | 7860 | `make dashboard` |
| All three | - | `make serve` |

## ML Engineering

```bash
make train               # Train ensemble + register in MLflow
make validate            # Golden set validation (pass/fail gate)
make promote-staging     # Promote model (validates first)
make promote-production  # Promote to production (validates first)
make models-list         # List model versions with metrics
```

## Full Pipeline

```bash
uv run python scripts/collect_data.py                      # 1. Collect ~100k items
uv run python scripts/generate_labels.py --sample-size 10000  # 2. Label via Gemini 3.0 Flash
uv run python scripts/finetune_embeddings.py               # 3. Fine-tune e5-base
uv run python scripts/tune_ensemble.py --n-trials 60       # 4. Optuna hyperparameter search
make train                                                 # 5. Train ensemble
make classify-all                                          # 6. Classify all items
```

## Docker

```bash
make docker-build    # Build image
make docker-run      # Run with docker-compose (models mounted as volume)
```

## Tests

```bash
make test       # 52 unit tests
make lint       # Code quality check (ruff)
```

## Documentation

- [`docs/architecture.pdf`](https://github.com/Wesley-S-Alves/Hazmat/blob/main/docs/architecture.pdf) — Architecture document (system design, MCP, production considerations)
- [`docs/usage_guide.pdf`](https://github.com/Wesley-S-Alves/Hazmat/blob/main/docs/usage_guide.pdf) — Detailed usage guide with screenshots
- [`notebooks/eda_hazmat.ipynb`](https://github.com/Wesley-S-Alves/Hazmat/blob/main/notebooks/eda_hazmat.ipynb) — Exploratory data analysis notebook
- [`prompts/prompts_used.md`](https://github.com/Wesley-S-Alves/Hazmat/blob/main/prompts/prompts_used.md) — All LLM prompts documented

## Tech Stack

Python 3.11+ | sentence-transformers (fine-tuned e5-base) | XGBoost + LightGBM + RF | Optuna | MLflow | Gemini 3.0 Flash | FastAPI | Docker | pytest | GitHub Actions | Gradio | SHAP | Platt Scaling

## Explainability (SHAP)

Classifications include human-readable explanations powered by SHAP:

```
[HAZ] conf=0.95 | Gasolina combustivel 5 litros
       Hazmat (conf: 0.96) due to: semantic analysis (+0.66), matched keywords: combustivel, gasolina

[ok ] conf=0.93 | Camiseta algodao 100% organico
       Non-hazmat (conf: 0.03) due to: semantic analysis (-0.62)

[HAZ] conf=0.95 | Inseticida spray mata barata 300ml
       Hazmat (conf: 0.96) due to: semantic analysis (+0.72), matched keywords: inseticida, mata barata, spray
```

Probabilities are calibrated via **Platt Scaling** (logistic calibration on validation set).
