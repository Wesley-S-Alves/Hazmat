#!/usr/bin/env python3
"""Train the hazmat ensemble with full MLflow experiment tracking.

Logs everything: params, metrics, artifacts, model, feature config.
Registers the model in MLflow Model Registry for staging/production promotion.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --run-name "experiment-v2"
"""

import os

# Must be set before ANY native library import (XGBoost, LightGBM use OpenMP)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FeatureBuilder
from src.model import HazmatEnsemble
from src.observability import DriftDetector, setup_logging

setup_logging()
logger = logging.getLogger("hazmat.train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS = str(PROJECT_ROOT / "mlruns")
MODEL_REGISTRY_NAME = "hazmat-classifier"


def load_training_data() -> pd.DataFrame:
    """Load and merge training labels from all sources."""
    frames = []

    for name, parquet, csv in [
        ("LLM", "data/processed/labels_llm.parquet", "data/processed/labels_llm.csv"),
        ("manual", "data/processed/labels_manual.parquet", "data/processed/labels_manual.csv"),
    ]:
        p, c = Path(parquet), Path(csv)
        if p.exists():
            df = pd.read_parquet(p)
            df["label_source"] = name.lower()
            frames.append(df)
            logger.info("Loaded %d %s labels from %s", len(df), name, p)
        elif c.exists():
            df = pd.read_csv(c)
            df["label_source"] = name.lower()
            frames.append(df)
            logger.info("Loaded %d %s labels from %s (CSV fallback)", len(df), name, c)

    if not frames:
        logger.error("No training labels found! Run generate_labels.py first.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("label_source", ascending=False)  # manual first
    combined = combined.drop_duplicates(subset=["item_id"], keep="first")

    hazmat_count = combined["is_hazmat"].sum()
    logger.info("Training set: %d items (%d hazmat, %d non-hazmat, %.1f%% hazmat)",
                len(combined), hazmat_count, len(combined) - hazmat_count,
                100 * hazmat_count / len(combined))
    return combined


def compute_detailed_metrics(y_true, y_pred, y_proba, prefix="") -> dict:
    """Compute comprehensive metrics for logging."""
    p = prefix + "_" if prefix else ""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        f"{p}f1": f1_score(y_true, y_pred),
        f"{p}precision": precision_score(y_true, y_pred),
        f"{p}recall": recall_score(y_true, y_pred),
        f"{p}roc_auc": roc_auc_score(y_true, y_proba),
        f"{p}true_positives": int(tp),
        f"{p}true_negatives": int(tn),
        f"{p}false_positives": int(fp),
        f"{p}false_negatives": int(fn),
        f"{p}false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        f"{p}false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        f"{p}accuracy": float((tp + tn) / (tp + tn + fp + fn)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    args = parser.parse_args()

    run_name = args.run_name or f"train-{time.strftime('%Y%m%d-%H%M%S')}"

    logger.info("=" * 60)
    logger.info("HAZMAT ENSEMBLE TRAINING (MLflow tracked)")
    logger.info("=" * 60)

    # Load data
    train_df = load_training_data()
    if train_df.empty:
        return

    for col in ["title", "description", "category_id"]:
        if col not in train_df.columns:
            train_df[col] = ""

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(MODEL_REGISTRY_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run: %s (ID: %s)", run_name, run_id)

        # --- Log LLM prompts via Prompt Registry ---
        from src.llm_fallback import SYSTEM_PROMPT
        import hashlib
        prompt_hash = hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8]
        mlflow.log_param("llm_prompt_hash", prompt_hash)
        mlflow.log_param("llm_model", "gemini-flash-latest")

        # Register prompt in MLflow Prompt Registry (creates new version if changed)
        try:
            prompt = mlflow.genai.register_prompt(
                name="hazmat-system-prompt",
                template=SYSTEM_PROMPT,
                commit_message=f"Training run: {run_name}",
                tags={"model": "gemini-flash-latest", "temperature": "0"},
            )
            mlflow.log_param("llm_prompt_version", prompt.version)
            logger.info("Prompt registered: hazmat-system-prompt v%d (hash: %s)",
                        prompt.version, prompt_hash)
        except Exception as e:
            logger.warning("Prompt registry failed (non-critical): %s", e)
            # Fallback: save as artifact
            prompt_path = PROJECT_ROOT / "data" / "output" / "llm_system_prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, "w") as f:
                f.write(SYSTEM_PROMPT)
            mlflow.log_artifact(str(prompt_path), artifact_path="prompts")

        # Also log the prompts_used.md documentation
        prompts_doc = PROJECT_ROOT / "prompts" / "prompts_used.md"
        if prompts_doc.exists():
            mlflow.log_artifact(str(prompts_doc), artifact_path="prompts")

        # --- Log data lineage ---
        mlflow.log_params({
            "training_samples": len(train_df),
            "hazmat_count": int(train_df["is_hazmat"].sum()),
            "hazmat_ratio": round(float(train_df["is_hazmat"].mean()), 4),
            "label_sources": str(train_df["label_source"].value_counts().to_dict()),
            "n_categories": int(train_df["domain_id"].nunique()) if "domain_id" in train_df.columns else 0,
        })

        # --- Build features ---
        logger.info("Building features...")
        feature_start = time.time()
        feature_builder = FeatureBuilder()
        X = feature_builder.fit_transform(train_df)
        y = train_df["is_hazmat"].astype(int).values
        feature_time = time.time() - feature_start

        mlflow.log_params({
            "feature_dims": X.shape[1],
            "embedding_model": feature_builder.embedding_model_name,
            "embedding_dims": 768,
            "keyword_dims": X.shape[1] - 768 - 1,
            "feature_build_time_s": round(feature_time, 1),
        })

        # --- Train ensemble ---
        logger.info("Training ensemble...")
        train_start = time.time()
        ensemble = HazmatEnsemble()

        # Log all hyperparameters
        mlflow.log_params({
            "xgb_n_estimators": ensemble.xgboost.n_estimators,
            "xgb_max_depth": ensemble.xgboost.max_depth,
            "xgb_learning_rate": ensemble.xgboost.learning_rate,
            "lgb_n_estimators": ensemble.lightgbm.n_estimators,
            "lgb_max_depth": ensemble.lightgbm.max_depth,
            "lgb_learning_rate": ensemble.lightgbm.learning_rate,
            "rf_n_estimators": ensemble.random_forest.n_estimators,
            "rf_max_depth": ensemble.random_forest.max_depth,
            "ensemble_weights": str(ensemble.weights),
            "decision_threshold": ensemble.threshold,
        })

        # Set feature names for SHAP explanations
        n_kw = X.shape[1] - 768 - 1  # total features - embeddings - category
        kw_names = [f"kw_{t}" for t in feature_builder._keyword_terms]
        # Pad if keyword features > keyword terms (derived features)
        while len(kw_names) < n_kw:
            kw_names.append(f"kw_derived_{len(kw_names)}")
        kw_names = kw_names[:n_kw]  # Trim if needed

        feature_names = (
            [f"emb_{i}" for i in range(768)]
            + kw_names
            + ["category"]
        )
        assert len(feature_names) == X.shape[1], f"Feature names mismatch: {len(feature_names)} vs {X.shape[1]}"
        ensemble.set_feature_names(feature_names)

        metrics = ensemble.train(X, y, track_mlflow=False)  # We handle MLflow ourselves
        train_time = time.time() - train_start

        mlflow.log_metric("train_time_s", round(train_time, 1))

        # --- Log per-model metrics ---
        all_metrics = {}
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[f"{model_name}_{metric_name}"] = value

        mlflow.log_metrics(all_metrics)

        # --- Log ensemble detailed metrics ---
        from sklearn.model_selection import train_test_split
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        val_proba = ensemble.predict_proba(X_val)
        val_pred = (val_proba >= ensemble.threshold).astype(int)
        detailed = compute_detailed_metrics(y_val, val_pred, val_proba, prefix="val")
        mlflow.log_metrics(detailed)

        # --- Save artifacts ---
        models_dir = PROJECT_ROOT / "models"
        feature_builder.save()
        ensemble.save()

        # --- Save drift baseline ---
        baseline_path = models_dir / "drift_baseline.json"
        DriftDetector.save_baseline(X, y, output_path=baseline_path)
        mlflow.log_artifact(str(baseline_path), artifact_path="models")

        # Log model artifacts to MLflow
        mlflow.log_artifacts(str(models_dir), artifact_path="models")

        # Log config files
        config_path = PROJECT_ROOT / "data" / "output" / "best_ensemble_config.json"
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")

        keywords_path = PROJECT_ROOT / "configs" / "hazmat_keywords.yaml"
        if keywords_path.exists():
            mlflow.log_artifact(str(keywords_path), artifact_path="config")

        # --- Log classification report as artifact ---
        report = classification_report(y_val, val_pred, target_names=["Non-Hazmat", "Hazmat"])
        report_path = PROJECT_ROOT / "data" / "output" / "classification_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Run: {run_name} (ID: {run_id})\n")
            f.write(f"Threshold: {ensemble.threshold}\n\n")
            f.write(report)
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_val, val_pred)}\n")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # --- Log training data hash for reproducibility ---
        data_hash = pd.util.hash_pandas_object(train_df[["item_id", "is_hazmat"]]).sum()
        mlflow.log_param("data_hash", str(data_hash))

        # --- Register model in MLflow Model Registry ---
        try:
            # Log sklearn-compatible model for registry
            mlflow.sklearn.log_model(
                sk_model=ensemble.xgboost,
                artifact_path="model",
            )
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
            logger.info("Model registered: %s version %s", MODEL_REGISTRY_NAME, mv.version)

            # Tag the version
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME, mv.version, "f1", str(round(metrics["ensemble"]["f1"], 4))
            )
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME, mv.version, "fn_rate",
                str(round(metrics["ensemble"]["false_negative_rate"] * 100, 1))
            )
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME, mv.version, "threshold", str(ensemble.threshold)
            )
        except Exception as e:
            logger.warning("Model registration failed: %s", e)

        # --- Summary ---
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("  Run:              %s", run_name)
        logger.info("  MLflow Run ID:    %s", run_id)
        logger.info("  XGBoost F1:       %.4f", metrics["xgboost"]["f1"])
        logger.info("  LightGBM F1:      %.4f", metrics["lightgbm"]["f1"])
        logger.info("  Random Forest F1: %.4f", metrics["random_forest"]["f1"])
        logger.info("  Ensemble F1:      %.4f", metrics["ensemble"]["f1"])
        logger.info("  ROC AUC:          %.4f", detailed["val_roc_auc"])
        logger.info("  False Negatives:  %d (%.1f%%)",
                     metrics["ensemble"]["false_negatives"],
                     metrics["ensemble"]["false_negative_rate"] * 100)
        logger.info("  Artifacts:        %s", models_dir)
        logger.info("=" * 60)
        logger.info("")
        logger.info("View in MLflow UI:  mlflow ui --backend-store-uri %s", MLFLOW_DB)


if __name__ == "__main__":
    main()
