#!/usr/bin/env python3
"""Validate a model against a golden test set before promotion.

Loads the latest model from MLflow registry, runs it against a curated
validation set, and reports pass/fail based on minimum metric thresholds.

Usage:
    python scripts/validate_model.py
    python scripts/validate_model.py --min-f1 0.90 --max-fn-rate 0.06
    python scripts/validate_model.py --version 3  # validate specific version
"""

import os
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FeatureBuilder
from src.model import HazmatEnsemble
from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.validate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MODEL_REGISTRY_NAME = "hazmat-classifier"

# Known items with verified labels (golden set)
GOLDEN_SET = [
    # Hazmat items
    {"title": "Gasolina comum 5 litros", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Thinner para diluicao de tintas 500ml", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Bateria Litio 18650 3.7V recarregavel", "is_hazmat": True, "class": "lithium_battery"},
    {"title": "Alcool etilico 70% 1 litro", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Extintor de incendio po ABC 4kg", "is_hazmat": True, "class": "compressed_gas"},
    {"title": "Soda caustica 1kg para desentupir", "is_hazmat": True, "class": "corrosive"},
    {"title": "Inseticida spray mata tudo 300ml", "is_hazmat": True, "class": "toxic"},
    {"title": "Agua sanitaria 5 litros hipoclorito", "is_hazmat": True, "class": "corrosive"},
    {"title": "Removedor de esmalte acetona pura 100ml", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Tinta spray preto fosco 400ml", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Oleo motor sintetico 5W30 1 litro", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Herbicida glifosato 1 litro", "is_hazmat": True, "class": "toxic"},
    {"title": "Cola instantanea super bonder 5g", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Fluido de freio DOT4 500ml", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Querosene 1 litro para lampiao", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Raticida veneno para rato granulado", "is_hazmat": True, "class": "toxic"},
    {"title": "Desodorante aerossol masculino 150ml", "is_hazmat": True, "class": "compressed_gas"},
    {"title": "Pilha recarregavel AA Duracell", "is_hazmat": True, "class": "lithium_battery"},
    {"title": "Resina epoxi transparente bicomponente", "is_hazmat": True, "class": "flammable_liquid"},
    {"title": "Permanganato de potassio 100g", "is_hazmat": True, "class": "oxidizer"},

    # Non-hazmat items
    {"title": "Camiseta algodao azul masculina GG", "is_hazmat": False, "class": "clothing"},
    {"title": "Travesseiro viscoelastico NASA", "is_hazmat": False, "class": "home"},
    {"title": "Quebra-cabeca 1000 pecas paisagem", "is_hazmat": False, "class": "toys"},
    {"title": "Cabo HDMI 2.1 4K 2 metros", "is_hazmat": False, "class": "electronics_accessory"},
    {"title": "Bolsa feminina couro sintetico", "is_hazmat": False, "class": "fashion"},
    {"title": "Tapete sala 2x3 metros bege", "is_hazmat": False, "class": "home"},
    {"title": "Livro Harry Potter capa dura", "is_hazmat": False, "class": "books"},
    {"title": "Oculos de sol polarizado UV400", "is_hazmat": False, "class": "fashion"},
    {"title": "Toalha de banho algodao felpuda", "is_hazmat": False, "class": "home"},
    {"title": "Panela inox 5 litros com tampa", "is_hazmat": False, "class": "kitchen"},
    {"title": "Cinto couro masculino fivela aco", "is_hazmat": False, "class": "fashion"},
    {"title": "Luminaria LED mesa escritorio", "is_hazmat": False, "class": "electronics_accessory"},
    {"title": "Organizador gaveta plastico", "is_hazmat": False, "class": "home"},
    {"title": "Mochila escolar infantil", "is_hazmat": False, "class": "fashion"},
    {"title": "Frigideira antiaderente 28cm", "is_hazmat": False, "class": "kitchen"},

    # Edge cases (tricky ones)
    {"title": "Acido hialuronico serum facial 30ml", "is_hazmat": False, "class": "cosmetics_safe"},
    {"title": "Creme depilatório Veet 100ml", "is_hazmat": False, "class": "cosmetics_safe"},
    {"title": "Carregador USB para bateria 18650", "is_hazmat": False, "class": "electronics_accessory"},
    {"title": "Vela aromatica decorativa vanilla", "is_hazmat": True, "class": "flammable_solid"},
    {"title": "Smartphone Samsung Galaxy bateria litio", "is_hazmat": True, "class": "lithium_battery"},
]


def validate(min_f1: float = 0.85, max_fn_rate: float = 0.08, model_version: int | None = None):
    """Run validation against golden set.

    Returns:
        dict with validation results and pass/fail status
    """
    logger.info("=" * 60)
    logger.info("MODEL VALIDATION")
    logger.info("=" * 60)

    # Load model
    models_dir = PROJECT_ROOT / "models"
    feature_builder = FeatureBuilder()
    feature_builder.load(models_dir)
    ensemble = HazmatEnsemble()
    ensemble.load(models_dir)

    # Build golden set DataFrame
    golden_df = pd.DataFrame(GOLDEN_SET)
    golden_df["description"] = ""
    golden_df["category_id"] = ""
    golden_df["domain_id"] = ""
    golden_df["item_id"] = [f"golden_{i}" for i in range(len(golden_df))]

    # Predict
    start = time.time()
    X = feature_builder.transform(golden_df, update_cache=False)
    detailed = ensemble.predict_detailed(X)
    elapsed = time.time() - start

    predictions = detailed["predictions"]
    confidence = detailed["confidence"]
    ensemble_proba = detailed["ensemble_proba"]

    y_true = golden_df["is_hazmat"].astype(int).values
    y_pred = predictions

    # Metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Per-item results
    logger.info("\nPer-item results:")
    errors = []
    for i, row in golden_df.iterrows():
        pred_hazmat = bool(predictions[i])
        correct = pred_hazmat == row["is_hazmat"]
        tag = "OK" if correct else "FAIL"
        label = "HAZ" if pred_hazmat else "ok "
        expected = "HAZ" if row["is_hazmat"] else "ok "

        if not correct:
            error_type = "FALSE_NEGATIVE" if row["is_hazmat"] and not pred_hazmat else "FALSE_POSITIVE"
            errors.append({
                "title": row["title"],
                "expected": row["is_hazmat"],
                "predicted": pred_hazmat,
                "error_type": error_type,
                "confidence": float(confidence[i]),
                "ensemble_proba": float(ensemble_proba[i]),
                "class": row["class"],
            })
            logger.info("  [%s] [%s→%s] conf=%.3f proba=%.3f | %s (%s)",
                        tag, expected, label, confidence[i], ensemble_proba[i],
                        row["title"][:60], row["class"])
        else:
            logger.info("  [%s]  [%s]    conf=%.3f proba=%.3f | %s",
                        tag, label, confidence[i], ensemble_proba[i],
                        row["title"][:60])

    # Summary
    passed_f1 = f1 >= min_f1
    passed_fn = fn_rate <= max_fn_rate
    passed = passed_f1 and passed_fn

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION %s", "PASSED" if passed else "FAILED")
    logger.info("=" * 60)
    logger.info("  Golden set size:    %d items", len(golden_df))
    logger.info("  Inference time:     %.1fms (%.1fms/item)", elapsed * 1000, elapsed * 1000 / len(golden_df))
    logger.info("  F1:                 %.4f %s (min: %.2f)", f1, "PASS" if passed_f1 else "FAIL", min_f1)
    logger.info("  Precision:          %.4f", precision)
    logger.info("  Recall:             %.4f", recall)
    logger.info("  False negatives:    %d (%.1f%%) %s (max: %.0f%%)",
                fn, fn_rate * 100, "PASS" if passed_fn else "FAIL", max_fn_rate * 100)
    logger.info("  False positives:    %d", fp)
    logger.info("  Errors:             %d / %d", len(errors), len(golden_df))
    if errors:
        logger.info("  Error details:")
        for e in errors:
            logger.info("    [%s] %s (conf=%.3f, class=%s)",
                        e["error_type"], e["title"][:50], e["confidence"], e["class"])
    logger.info("=" * 60)

    # Save results
    results = {
        "passed": passed,
        "golden_set_size": len(golden_df),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_negatives": int(fn),
        "false_negative_rate": round(fn_rate, 4),
        "false_positives": int(fp),
        "errors": errors,
        "thresholds": {"min_f1": min_f1, "max_fn_rate": max_fn_rate},
        "inference_time_ms": round(elapsed * 1000, 1),
    }

    output_path = PROJECT_ROOT / "data" / "output" / "validation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Log to MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_DB)
        mlflow.set_experiment(MODEL_REGISTRY_NAME)
        with mlflow.start_run(run_name=f"validation-{time.strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_metrics({
                "golden_f1": f1,
                "golden_precision": precision,
                "golden_recall": recall,
                "golden_fn_rate": fn_rate,
                "golden_fp_count": int(fp),
                "golden_fn_count": int(fn),
                "golden_errors": len(errors),
                "golden_passed": 1 if passed else 0,
            })
            mlflow.log_artifact(str(output_path), artifact_path="validation")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-f1", type=float, default=0.85, help="Minimum F1 to pass")
    parser.add_argument("--max-fn-rate", type=float, default=0.08, help="Maximum false negative rate")
    parser.add_argument("--version", type=int, default=None, help="Model version to validate")
    args = parser.parse_args()

    results = validate(min_f1=args.min_f1, max_fn_rate=args.max_fn_rate, model_version=args.version)

    # Exit code: 0 if passed, 1 if failed (useful for CI/CD)
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
