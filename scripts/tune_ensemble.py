#!/usr/bin/env python3
"""Hyperparameter tuning for the hazmat ensemble using Optuna.

Bayesian optimization (TPE) with early pruning. Much faster than grid search.

Usage:
    python scripts/tune_ensemble.py
    python scripts/tune_ensemble.py --n-trials 100
    python scripts/tune_ensemble.py --n-trials 30 --quick
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FeatureBuilder
from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.tune")


def load_data():
    """Load training data and build features."""
    llm_path = Path("data/processed/labels_llm.parquet")
    if not llm_path.exists():
        logger.error("No labels found at %s", llm_path)
        return None, None

    df = pd.read_parquet(llm_path)
    for col in ["title", "description", "category_id"]:
        if col not in df.columns:
            df[col] = ""

    logger.info("Loaded %d labeled items (%.1f%% hazmat)",
                len(df), 100 * df["is_hazmat"].mean())

    fb = FeatureBuilder()
    X = fb.fit_transform(df)
    y = df["is_hazmat"].astype(int).values
    return X, y


def create_objective(X, y, n_folds=3):
    """Create an Optuna objective function that tunes ALL params jointly."""

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    def objective(trial: optuna.Trial) -> float:
        # XGBoost params (shallow, fast)
        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.05, 0.2, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.7, 1.0),
        }

        # LightGBM params (shallow, fast)
        lgb_params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("lgb_max_depth", 3, 6),
            "learning_rate": trial.suggest_float("lgb_learning_rate", 0.05, 0.2, log=True),
            "subsample": trial.suggest_float("lgb_subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.7, 1.0),
        }

        # Random Forest params (shallow, fast)
        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 5, 12),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 3, 10),
        }

        # Ensemble weights
        w_xgb = trial.suggest_float("w_xgb", 0.1, 0.6)
        w_lgb = trial.suggest_float("w_lgb", 0.1, 0.6)
        w_rf = trial.suggest_float("w_rf", 0.05, 0.4)

        # Decision threshold
        threshold = trial.suggest_float("threshold", 0.3, 0.55)

        fold_f1s = []
        fold_recalls = []
        fold_fn_rates = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            scale = n_neg / n_pos if n_pos > 0 else 1.0

            # Train
            xgb = XGBClassifier(
                **xgb_params, scale_pos_weight=scale,
                eval_metric="logloss", random_state=42, nthread=1, n_jobs=1,
            )
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            lgb = LGBMClassifier(
                **lgb_params, scale_pos_weight=scale,
                random_state=42, n_jobs=1, verbose=-1,
            )
            lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            rf = RandomForestClassifier(
                **rf_params, class_weight="balanced",
                random_state=42, n_jobs=1,
            )
            rf.fit(X_train, y_train)

            # Ensemble
            total_w = w_xgb + w_lgb + w_rf
            proba = (
                (w_xgb / total_w) * xgb.predict_proba(X_val)[:, 1]
                + (w_lgb / total_w) * lgb.predict_proba(X_val)[:, 1]
                + (w_rf / total_w) * rf.predict_proba(X_val)[:, 1]
            )
            pred = (proba >= threshold).astype(int)

            f1 = f1_score(y_val, pred)
            recall = recall_score(y_val, pred)
            fn = ((y_val == 1) & (pred == 0)).sum()
            fn_rate = fn / y_val.sum() if y_val.sum() > 0 else 0

            fold_f1s.append(f1)
            fold_recalls.append(recall)
            fold_fn_rates.append(fn_rate)

            # Early pruning: report intermediate value after each fold
            trial.report(np.mean(fold_f1s), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_f1 = np.mean(fold_f1s)
        mean_recall = np.mean(fold_recalls)
        mean_fn_rate = np.mean(fold_fn_rates)

        # Store extra metrics as user attributes
        trial.set_user_attr("recall", mean_recall)
        trial.set_user_attr("fn_rate", mean_fn_rate)
        trial.set_user_attr("f1_std", np.std(fold_f1s))

        # Objective: F1 with penalty for false negatives
        score = mean_f1 - 0.1 * mean_fn_rate
        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Tune hazmat ensemble with Optuna")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds for tuning")
    args = parser.parse_args()

    logger.info("Loading data and features...")
    X, y = load_data()
    if X is None:
        return

    logger.info("=" * 70)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("  Trials:  %d", args.n_trials)
    logger.info("  CV folds: %d", args.n_folds)
    logger.info("  Timeout:  %s", f"{args.timeout}s" if args.timeout else "None")
    logger.info("  Objective: F1 - 0.1 * FN_rate (penalize missed hazmat)")
    logger.info("=" * 70)

    objective = create_objective(X, y, n_folds=args.n_folds)

    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="hazmat-ensemble-tuning",
    )

    start_time = time.time()

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            elapsed = time.time() - start_time
            recall = trial.user_attrs.get("recall", 0)
            fn_rate = trial.user_attrs.get("fn_rate", 0)
            f1_std = trial.user_attrs.get("f1_std", 0)

            logger.info(
                "Trial %d/%d | Score=%.4f | Recall=%.4f | FN_rate=%.1f%% | "
                "Best=%.4f | Elapsed: %.0fs",
                trial.number + 1, args.n_trials,
                trial.value, recall, fn_rate * 100,
                study.best_value, elapsed,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info("Trial %d/%d | PRUNED (poor early fold)", trial.number + 1, args.n_trials)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[log_trial],
    )

    total_time = time.time() - start_time

    # Results
    best = study.best_trial
    logger.info("")
    logger.info("=" * 70)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time:    %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Trials:        %d completed, %d pruned",
                len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
    logger.info("Best score:    %.4f (F1 - 0.1*FN_rate)", best.value)
    logger.info("Best recall:   %.4f", best.user_attrs.get("recall", 0))
    logger.info("Best FN rate:  %.1f%%", best.user_attrs.get("fn_rate", 0) * 100)
    logger.info("")

    # Extract best params by category
    bp = best.params
    best_config = {
        "xgboost": {k.replace("xgb_", ""): v for k, v in bp.items() if k.startswith("xgb_")},
        "lightgbm": {k.replace("lgb_", ""): v for k, v in bp.items() if k.startswith("lgb_")},
        "random_forest": {k.replace("rf_", ""): v for k, v in bp.items() if k.startswith("rf_")},
        "ensemble_weights": [bp["w_xgb"], bp["w_lgb"], bp["w_rf"]],
        "threshold": bp["threshold"],
        "score": best.value,
        "recall": best.user_attrs.get("recall", 0),
        "fn_rate": best.user_attrs.get("fn_rate", 0),
        "total_trials": len(study.trials),
        "total_time_s": round(total_time, 1),
    }

    logger.info("Best XGBoost:  %s", json.dumps(best_config["xgboost"], indent=2))
    logger.info("Best LightGBM: %s", json.dumps(best_config["lightgbm"], indent=2))
    logger.info("Best RF:       %s", json.dumps(best_config["random_forest"], indent=2))
    logger.info("Weights:       XGB=%.3f LGB=%.3f RF=%.3f",
                bp["w_xgb"], bp["w_lgb"], bp["w_rf"])
    logger.info("Threshold:     %.3f", bp["threshold"])

    # Save config
    config_path = Path("data/output/best_ensemble_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Config saved to %s", config_path)

    # Log best to MLflow
    try:
        mlflow.set_experiment("hazmat-tuning")
        with mlflow.start_run(run_name="optuna-best"):
            mlflow.log_params(bp)
            mlflow.log_metrics({
                "best_score": best.value,
                "best_recall": best.user_attrs.get("recall", 0),
                "best_fn_rate": best.user_attrs.get("fn_rate", 0),
                "total_trials": len(study.trials),
                "tuning_time_s": total_time,
            })
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    # Top 5 trials
    logger.info("")
    logger.info("Top 5 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    for i, t in enumerate(completed[:5]):
        logger.info("  #%d | Score=%.4f | Recall=%.4f | FN=%.1f%% | threshold=%.3f",
                     i + 1, t.value,
                     t.user_attrs.get("recall", 0),
                     t.user_attrs.get("fn_rate", 0) * 100,
                     t.params.get("threshold", 0))

    logger.info("")
    logger.info("To retrain with best params: update src/model.py or use data/output/best_ensemble_config.json")


if __name__ == "__main__":
    main()
