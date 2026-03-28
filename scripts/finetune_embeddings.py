#!/usr/bin/env python3
"""Fine-tune multilingual-e5-base on hazmat classification labels.

Uses the labeled data to teach the embedding model that hazmat items
should cluster together and non-hazmat items should be distant from them.

This creates a specialized embedding model saved to models/e5-hazmat/
that replaces the generic multilingual-e5-base in the feature pipeline.

Usage:
    python scripts/finetune_embeddings.py
    python scripts/finetune_embeddings.py --epochs 5 --batch-size 32
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.finetune")


def build_pairs(df: pd.DataFrame) -> list[dict]:
    """Build training pairs from labeled data.

    Creates pairs of (text, text, label) where:
    - label=1 if both are hazmat or both are non-hazmat (similar)
    - label=0 if one is hazmat and the other is not (dissimilar)
    """
    from datasets import Dataset

    hazmat = df[df["is_hazmat"] == True].reset_index(drop=True)
    non_hazmat = df[df["is_hazmat"] == False].reset_index(drop=True)

    logger.info("Building pairs: %d hazmat, %d non-hazmat items", len(hazmat), len(non_hazmat))

    pairs = []
    rng = pd.np.random.RandomState(42) if hasattr(pd, "np") else __import__("numpy").random.RandomState(42)

    n_pairs_per_type = min(len(hazmat), len(non_hazmat), 5000)

    # Positive pairs: hazmat-hazmat
    idx = rng.choice(len(hazmat), size=(n_pairs_per_type, 2), replace=True)
    for i, j in idx:
        if i != j:
            pairs.append({
                "sentence1": _make_text(hazmat.iloc[i]),
                "sentence2": _make_text(hazmat.iloc[j]),
                "label": 1.0,
            })

    # Positive pairs: non_hazmat-non_hazmat
    idx = rng.choice(len(non_hazmat), size=(n_pairs_per_type, 2), replace=True)
    for i, j in idx:
        if i != j:
            pairs.append({
                "sentence1": _make_text(non_hazmat.iloc[i]),
                "sentence2": _make_text(non_hazmat.iloc[j]),
                "label": 1.0,
            })

    # Negative pairs: hazmat-non_hazmat
    idx_h = rng.choice(len(hazmat), size=n_pairs_per_type * 2, replace=True)
    idx_n = rng.choice(len(non_hazmat), size=n_pairs_per_type * 2, replace=True)
    for i, j in zip(idx_h, idx_n):
        pairs.append({
            "sentence1": _make_text(hazmat.iloc[i]),
            "sentence2": _make_text(non_hazmat.iloc[j]),
            "label": 0.0,
        })

    rng.shuffle(pairs)
    logger.info("Created %d training pairs (%.0f%% positive)",
                len(pairs), 100 * sum(p["label"] for p in pairs) / len(pairs))

    return Dataset.from_list(pairs)


def build_triplets(df: pd.DataFrame) -> "Dataset":
    """Build triplet training data: (anchor, positive, negative).

    More effective than pairs for contrastive learning.
    """
    from datasets import Dataset
    import numpy as np

    rng = np.random.RandomState(42)

    hazmat = df[df["is_hazmat"] == True].reset_index(drop=True)
    non_hazmat = df[df["is_hazmat"] == False].reset_index(drop=True)

    logger.info("Building triplets: %d hazmat, %d non-hazmat items", len(hazmat), len(non_hazmat))

    triplets = []
    n_triplets = min(len(hazmat) * 3, len(non_hazmat) * 3, 8000)

    # Hazmat anchors
    for _ in range(n_triplets // 2):
        anchor_idx, pos_idx = rng.choice(len(hazmat), size=2, replace=False)
        neg_idx = rng.choice(len(non_hazmat))
        triplets.append({
            "anchor": _make_text(hazmat.iloc[anchor_idx]),
            "positive": _make_text(hazmat.iloc[pos_idx]),
            "negative": _make_text(non_hazmat.iloc[neg_idx]),
        })

    # Non-hazmat anchors
    for _ in range(n_triplets // 2):
        anchor_idx, pos_idx = rng.choice(len(non_hazmat), size=2, replace=False)
        neg_idx = rng.choice(len(hazmat))
        triplets.append({
            "anchor": _make_text(non_hazmat.iloc[anchor_idx]),
            "positive": _make_text(non_hazmat.iloc[pos_idx]),
            "negative": _make_text(hazmat.iloc[neg_idx]),
        })

    rng.shuffle(triplets)
    logger.info("Created %d triplets", len(triplets))

    return Dataset.from_list(triplets)


def _make_text(row) -> str:
    """Create text representation of an item (same as feature pipeline)."""
    title = str(row.get("title", ""))
    desc = str(row.get("description", ""))[:300]
    text = title
    if desc and desc != "nan":
        text += " " + desc
    return f"query: {text}"


def build_eval_pairs(df: pd.DataFrame) -> BinaryClassificationEvaluator:
    """Build evaluator from labeled data."""
    import numpy as np

    sentences1 = []
    sentences2 = []
    labels = []

    hazmat = df[df["is_hazmat"] == True].reset_index(drop=True)
    non_hazmat = df[df["is_hazmat"] == False].reset_index(drop=True)

    rng = np.random.RandomState(99)
    n = min(200, len(hazmat), len(non_hazmat))

    # Positive: same class
    for _ in range(n):
        i, j = rng.choice(len(hazmat), size=2, replace=False)
        sentences1.append(_make_text(hazmat.iloc[i]))
        sentences2.append(_make_text(hazmat.iloc[j]))
        labels.append(1)

    # Negative: different class
    for _ in range(n):
        i = rng.choice(len(hazmat))
        j = rng.choice(len(non_hazmat))
        sentences1.append(_make_text(hazmat.iloc[i]))
        sentences2.append(_make_text(non_hazmat.iloc[j]))
        labels.append(0)

    return BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=labels,
        name="hazmat-eval",
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune E5 embeddings for hazmat classification")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (small for MPS memory)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="Base model to fine-tune")
    parser.add_argument("--output", type=str, default="models/e5-hazmat",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--loss", type=str, default="triplet",
                        choices=["triplet", "cosine"],
                        help="Loss function: triplet (TripletLoss) or cosine (CosineSimilarityLoss)")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Load labeled data
    labels_path = Path("data/processed/labels_llm.parquet")
    if not labels_path.exists():
        logger.error("No labels found. Run generate_labels.py first.")
        return

    df = pd.read_parquet(labels_path)
    logger.info("Loaded %d labeled items (%.1f%% hazmat)", len(df), 100 * df["is_hazmat"].mean())

    # Train/eval split
    train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["is_hazmat"])
    logger.info("Train: %d | Eval: %d", len(train_df), len(eval_df))

    # Build training data
    if args.loss == "triplet":
        train_dataset = build_triplets(train_df)
    else:
        train_dataset = build_pairs(train_df)

    # Build evaluator
    evaluator = build_eval_pairs(eval_df)

    # Detect device
    # MPS runs out of memory during backprop on E5-base, so we use CPU for fine-tuning
    # (MPS is fine for inference/encoding, but training needs too much memory)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("=" * 70)
    logger.info("FINE-TUNING EMBEDDING MODEL")
    logger.info("  Base model:  %s", args.model)
    logger.info("  Device:      %s", device)
    logger.info("  Loss:        %s", args.loss)
    logger.info("  Epochs:      %d", args.epochs)
    logger.info("  Batch size:  %d", args.batch_size)
    logger.info("  LR:          %s", args.lr)
    logger.info("  Train pairs: %d", len(train_dataset))
    logger.info("  Output:      %s", output_path)
    logger.info("=" * 70)

    # Load model
    model = SentenceTransformer(args.model, device=device)

    # Loss function
    if args.loss == "triplet":
        loss = losses.TripletLoss(model=model)
    else:
        loss = losses.CosineSimilarityLoss(model=model)

    # Training args
    # Gradient accumulation to simulate larger batch: effective_batch = batch_size * grad_accum
    grad_accum = max(1, 16 // args.batch_size)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        warmup_steps=0.1,
        fp16=False,  # MPS doesn't support fp16
        bf16=False,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="hazmat-eval_ap",
        dataloader_pin_memory=False,  # MPS doesn't support pin_memory
        seed=42,
    )

    # Train
    start = time.time()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset.select(range(min(500, len(train_dataset)))),
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    elapsed = time.time() - start

    logger.info("Training completed in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Save final model
    model.save(str(output_path))
    logger.info("Model saved to %s", output_path)

    # Evaluate
    logger.info("Running final evaluation...")
    eval_result = evaluator(model)
    logger.info("Evaluation results: %s", eval_result)

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("  Model saved to: %s", output_path)
    logger.info("  To use: update FeatureBuilder to load from '%s'", output_path)
    logger.info("  Then retrain the ensemble: python scripts/train_model.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
