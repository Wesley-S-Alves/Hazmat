#!/usr/bin/env python3
"""Promote a model version in MLflow Model Registry.

Workflow: None → Staging → Production
Validates the model against the golden set before promotion.

Usage:
    python scripts/promote_model.py --to staging           # latest → staging
    python scripts/promote_model.py --to production        # staging → production
    python scripts/promote_model.py --version 3 --to staging  # specific version
    python scripts/promote_model.py --list                 # list all versions
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.promote")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MODEL_REGISTRY_NAME = "hazmat-classifier"


def list_versions():
    """List all registered model versions."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    except Exception:
        logger.error("No model '%s' found in registry. Run train_model.py first.", MODEL_REGISTRY_NAME)
        return

    if not versions:
        logger.info("No versions registered for '%s'", MODEL_REGISTRY_NAME)
        return

    logger.info("=" * 70)
    logger.info("MODEL REGISTRY: %s", MODEL_REGISTRY_NAME)
    logger.info("=" * 70)
    logger.info("  %-8s %-12s %-10s %-10s %-20s", "Version", "Stage", "F1", "FN Rate", "Created")
    logger.info("  " + "-" * 64)

    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        tags = v.tags or {}
        f1 = tags.get("f1", "?")
        fn_rate = tags.get("fn_rate", "?")
        stage = v.current_stage if hasattr(v, "current_stage") else "None"
        # Get aliases
        aliases = []
        if hasattr(v, "aliases"):
            aliases = v.aliases
        alias_str = f" [{', '.join(aliases)}]" if aliases else ""
        created = str(v.creation_timestamp)[:10] if v.creation_timestamp else "?"
        logger.info("  %-8s %-12s %-10s %-10s %-20s%s",
                    v.version, stage, f1, fn_rate, created, alias_str)

    logger.info("=" * 70)


def promote(version: int | None, target_stage: str, skip_validation: bool = False):
    """Promote a model version to staging or production."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()

    # Find the version to promote
    if version is None:
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        if not versions:
            logger.error("No versions found. Run train_model.py first.")
            return False

        if target_stage == "production":
            # Promote staging → production: find version with staging alias
            try:
                mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "staging")
                version = int(mv.version)
            except Exception:
                # Fallback: look for tagged versions
                staging = [v for v in versions if (v.tags or {}).get("promoted_to") == "staging"]
                if not staging:
                    logger.error("No model in staging. Promote to staging first.")
                    return False
                version = int(staging[0].version)
        else:
            # Promote latest → staging
            version = max(int(v.version) for v in versions)

    logger.info("Promoting version %d to %s...", version, target_stage)

    # Validate before promotion
    if not skip_validation:
        logger.info("Running validation before promotion...")
        from scripts.validate_model import validate
        results = validate()
        if not results["passed"]:
            logger.error("Validation FAILED. Model not promoted.")
            logger.error("  F1: %.4f (min: %.2f)", results["f1"], results["thresholds"]["min_f1"])
            logger.error("  FN rate: %.1f%% (max: %.0f%%)",
                        results["false_negative_rate"] * 100, results["thresholds"]["max_fn_rate"] * 100)
            return False
        logger.info("Validation PASSED. Proceeding with promotion.")

    # Promote using aliases (MLflow 2.x way)
    try:
        alias = target_stage.lower()  # "staging" or "production"
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, alias, str(version))
        logger.info("Set alias '%s' on version %d", alias, version)
    except Exception:
        # Fallback: use stage transition (older MLflow)
        try:
            client.transition_model_version_stage(
                MODEL_REGISTRY_NAME, str(version),
                stage=target_stage.capitalize(),
                archive_existing_versions=(target_stage == "production"),
            )
        except Exception as e:
            logger.warning("Stage transition failed: %s", e)

    # Tag the version
    client.set_model_version_tag(MODEL_REGISTRY_NAME, str(version), "promoted_to", target_stage)

    logger.info("=" * 60)
    logger.info("MODEL PROMOTED")
    logger.info("  Model:   %s", MODEL_REGISTRY_NAME)
    logger.info("  Version: %d", version)
    logger.info("  Stage:   %s", target_stage)
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", choices=["staging", "production"], help="Target stage")
    parser.add_argument("--version", type=int, default=None, help="Version to promote")
    parser.add_argument("--list", action="store_true", help="List all versions")
    parser.add_argument("--skip-validation", action="store_true", help="Skip golden set validation")
    args = parser.parse_args()

    if args.list:
        list_versions()
        return

    if not args.to:
        parser.error("--to is required (staging or production)")

    success = promote(args.version, args.to, skip_validation=args.skip_validation)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
