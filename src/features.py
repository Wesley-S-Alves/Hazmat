"""Feature engineering with sentence embeddings + keyword indicators.

Uses multilingual-e5-base for semantic embeddings (GPU/MPS accelerated),
keyword binary features from YAML config, and category encoding.
Embeddings are cached in Parquet for reuse.
"""

import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

from src.keywords import load_keywords

logger = logging.getLogger("hazmat.features")

from src.compat import get_project_root

_ROOT = get_project_root()
MODELS_DIR = Path(os.environ.get("MODELS_DIR", _ROOT / "models"))
EMBEDDING_CACHE_PATH = Path(
    os.environ.get(
        "EMBEDDING_CACHE_PATH", _ROOT / "data" / "processed" / "embeddings_cache.parquet"
    )
)
FINETUNED_MODEL_PATH = MODELS_DIR / "e5-hazmat"
HF_MODEL_NAME = "WesleySAlves/e5-hazmat-classifier"
BASE_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768


def _resolve_embedding_model() -> str:
    """Resolve embedding model: local fine-tuned → HF Hub → base model."""
    # 1. Local fine-tuned model (fastest, no download)
    if FINETUNED_MODEL_PATH.exists() and (FINETUNED_MODEL_PATH / "config.json").exists():
        logger.info("Using local fine-tuned model: %s", FINETUNED_MODEL_PATH)
        return str(FINETUNED_MODEL_PATH)
    # 2. HF Hub fine-tuned model (downloads once, cached by sentence-transformers)
    logger.info("Local model not found. Using HF Hub model: %s", HF_MODEL_NAME)
    return HF_MODEL_NAME


def _get_device() -> str:
    """Detect best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FeatureBuilder:
    """Build features combining embeddings + keyword indicators + category encoding."""

    def __init__(self, embedding_model_name: str | None = None):
        self.embedding_model_name = embedding_model_name or _resolve_embedding_model()
        self._embedding_model = None
        self._embedding_cache = None  # Loaded on first use, stays in memory
        self.category_encoder = LabelEncoder()
        self._keyword_terms: list[str] = []
        self._hazard_class_names: list[str] = []
        self._fitted = False
        self.device = os.environ.get("HAZMAT_DEVICE", _get_device())
        self._load_keyword_config()

    def _load_keyword_config(self):
        """Load keyword terms from YAML for binary feature extraction."""
        try:
            config = load_keywords()
            hazard_classes = config.get("hazard_classes", {})
            all_terms = []
            class_names = []
            for class_name, class_data in hazard_classes.items():
                class_names.append(class_name)
                for kw in class_data.get("keywords", []):
                    all_terms.append(kw.lower())
            self._keyword_terms = sorted(set(all_terms))
            self._hazard_class_names = class_names
            logger.info(
                "Loaded %d keyword terms from %d hazard classes",
                len(self._keyword_terms),
                len(class_names),
            )
        except Exception as e:
            logger.warning("Could not load keywords config: %s", e)
            self._keyword_terms = []
            self._hazard_class_names = []

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(
                "Loading embedding model '%s' on device '%s'...",
                self.embedding_model_name,
                self.device,
            )
            start = time.time()

            # Disable SDPA on MPS — causes silent crashes on Apple Silicon
            if self.device == "mps":
                import os

                os.environ["TRANSFORMERS_NO_SDPA"] = "1"

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )
            logger.info("Embedding model loaded in %.1fs", time.time() - start)
        return self._embedding_model

    def _compute_embeddings(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Compute sentence embeddings with progress logging.

        Falls back to CPU if MPS/CUDA fails (known SDPA issues on Apple Silicon).
        """
        logger.info(
            "Computing embeddings for %d texts (batch_size=%d, device=%s)...",
            len(texts),
            batch_size,
            self.device,
        )
        start = time.time()

        # E5 models require "query: " prefix for best performance
        prefixed = [f"query: {t}" for t in texts]

        try:
            embeddings = self.embedding_model.encode(
                prefixed,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning("Encoding failed on %s: %s. Falling back to CPU.", self.device, e)
                self.device = "cpu"
                self._embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
                embeddings = self.embedding_model.encode(
                    prefixed,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
            else:
                raise

        elapsed = time.time() - start
        throughput = len(texts) / elapsed if elapsed > 0 else 0
        logger.info("Embeddings computed in %.1fs (%.1f items/s)", elapsed, throughput)
        return embeddings

    def _load_cache(self, cache_path: Path) -> None:
        """Load embedding cache into memory (once)."""
        if self._embedding_cache is not None:
            return  # Already loaded

        if cache_path.exists():
            cache_df = pd.read_parquet(cache_path)
            emb_cols = [c for c in cache_df.columns if c.startswith("emb_")]
            self._embedding_cache = {}
            # Normalize item_id to string for consistent lookup
            for idx in range(len(cache_df)):
                item_id = str(cache_df.iloc[idx]["item_id"])
                self._embedding_cache[item_id] = cache_df.iloc[idx][emb_cols].values.astype(
                    np.float32
                )
            logger.info("Loaded %d cached embeddings into memory", len(self._embedding_cache))
        else:
            self._embedding_cache = {}
            logger.info("No embedding cache found, starting fresh")

    def _get_or_compute_embeddings(
        self, df: pd.DataFrame, cache_path: Path | None = None, update_cache: bool = True
    ) -> np.ndarray:
        """Get embeddings from cache or compute them.

        Args:
            df: DataFrame with title, description, item_id columns
            cache_path: Path to embedding cache parquet
            update_cache: If False, skip cache read/write (fast path for single-item inference)
        """
        texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()

        # Fast path: skip cache entirely for real-time inference
        if not update_cache:
            return self._compute_embeddings(texts)

        cache_path = cache_path or EMBEDDING_CACHE_PATH
        item_ids = [
            str(x)
            for x in (df["item_id"].tolist() if "item_id" in df.columns else list(range(len(df))))
        ]

        # Load cache into memory (only on first call)
        self._load_cache(cache_path)

        # Identify which items need embedding
        to_compute_idx = []
        to_compute_texts = []
        embeddings = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)

        for i, item_id in enumerate(item_ids):
            if item_id in self._embedding_cache:
                embeddings[i] = self._embedding_cache[item_id]
            else:
                to_compute_idx.append(i)
                to_compute_texts.append(texts[i])

        logger.info("Cache hit: %d / %d items", len(df) - len(to_compute_idx), len(df))

        # Compute missing embeddings
        if to_compute_texts:
            new_embeddings = self._compute_embeddings(to_compute_texts)
            for j, idx in enumerate(to_compute_idx):
                embeddings[idx] = new_embeddings[j]

            # Update in-memory cache
            for j, idx in enumerate(to_compute_idx):
                self._embedding_cache[item_ids[idx]] = new_embeddings[j].astype(np.float32)

            # Persist to disk
            all_rows = []
            for iid, emb in self._embedding_cache.items():
                row = {"item_id": str(iid)}
                for k in range(EMBEDDING_DIM):
                    row[f"emb_{k}"] = float(emb[k])
                all_rows.append(row)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                pd.DataFrame(all_rows).to_parquet(cache_path, index=False)
                logger.info("Updated embedding cache: %d total items", len(self._embedding_cache))
            except Exception as e:
                logger.warning("Could not save embedding cache: %s", e)

        return embeddings

    def _keyword_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract binary keyword features + hazard class match counts."""
        text = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()

        features = []
        feature_names = []

        # Binary per-term features
        for term in self._keyword_terms:
            features.append(
                text.str.contains(term, na=False, regex=False).astype(np.float32).values
            )
            feature_names.append(f"kw_{term[:30]}")

        # Hazard class match count features
        config = load_keywords()
        hazard_classes = config.get("hazard_classes", {})
        for class_name, class_data in hazard_classes.items():
            keywords = [kw.lower() for kw in class_data.get("keywords", [])]
            match_count = np.zeros(len(df), dtype=np.float32)
            for kw in keywords:
                match_count += (
                    text.str.contains(kw, na=False, regex=False).astype(np.float32).values
                )
            features.append(match_count)
            feature_names.append(f"class_{class_name}_count")

        # Total keyword matches
        total_matches = np.zeros(len(df), dtype=np.float32)
        for term in self._keyword_terms:
            total_matches += (
                text.str.contains(term, na=False, regex=False).astype(np.float32).values
            )
        features.append(total_matches)
        feature_names.append("total_keyword_matches")

        result = np.column_stack(features) if features else np.zeros((len(df), 0))
        logger.info("Keyword features: %d dimensions", result.shape[1])
        return result

    def _category_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode category as integer feature."""
        domain_col = "domain_id" if "domain_id" in df.columns else "category_id"
        if domain_col not in df.columns:
            return np.zeros((len(df), 1), dtype=np.float32)

        values = df[domain_col].fillna("unknown").astype(str)

        if fit:
            all_values = pd.concat([values, pd.Series(["unknown"])], ignore_index=True)
            self.category_encoder.fit(all_values)

        known = set(self.category_encoder.classes_)
        values = values.map(lambda x: x if x in known else "unknown")
        encoded = self.category_encoder.transform(values).reshape(-1, 1).astype(np.float32)
        # Normalize to 0-1 range so it doesn't dominate embedding features
        n_classes = len(self.category_encoder.classes_)
        if n_classes > 1:
            encoded = encoded / (n_classes - 1)
        return encoded

    def fit_transform(self, df: pd.DataFrame, cache_path: Path | None = None) -> np.ndarray:
        """Fit and transform: embeddings + keywords + category.

        Args:
            df: DataFrame with columns: title, description, category_id/domain_id
            cache_path: Optional path for embedding cache

        Returns:
            Feature matrix (n_samples, embedding_dim + n_keyword_features + 1)
        """
        logger.info("Building features for %d items...", len(df))

        embeddings = self._get_or_compute_embeddings(df, cache_path)
        kw_features = self._keyword_features(df)
        cat_features = self._category_features(df, fit=True)

        features = np.hstack([embeddings, kw_features, cat_features])
        self._fitted = True
        logger.info(
            "Feature matrix shape: %s (embeddings=%d, keywords=%d, category=1)",
            features.shape,
            embeddings.shape[1],
            kw_features.shape[1],
        )
        return features

    def transform(
        self, df: pd.DataFrame, cache_path: Path | None = None, update_cache: bool = True
    ) -> np.ndarray:
        """Transform using fitted encoders.

        Args:
            df: DataFrame with title, description, category_id/domain_id
            cache_path: Path to embedding cache
            update_cache: If False, skip cache (fast path for real-time single-item inference)
        """
        if not self._fitted:
            raise RuntimeError("FeatureBuilder not fitted. Call fit_transform first.")

        embeddings = self._get_or_compute_embeddings(df, cache_path, update_cache=update_cache)
        kw_features = self._keyword_features(df)
        cat_features = self._category_features(df, fit=False)

        return np.hstack([embeddings, kw_features, cat_features])

    def save(self, path: Path | None = None) -> None:
        """Save fitted encoders to disk."""
        path = path or MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.category_encoder, path / "category_encoder.joblib")
        joblib.dump(
            {
                "embedding_model_name": self.embedding_model_name,
                "keyword_terms": self._keyword_terms,
                "hazard_class_names": self._hazard_class_names,
            },
            path / "feature_config.joblib",
        )
        logger.info("Feature config saved to %s", path)

    def load(self, path: Path | None = None) -> None:
        """Load fitted encoders from disk."""
        path = path or MODELS_DIR
        self.category_encoder = joblib.load(path / "category_encoder.joblib")
        config = joblib.load(path / "feature_config.joblib")
        saved_model = config["embedding_model_name"]
        # If saved path doesn't exist locally, resolve to HF Hub or default
        if saved_model and Path(saved_model).exists():
            self.embedding_model_name = saved_model
        else:
            self.embedding_model_name = _resolve_embedding_model()
            if self.embedding_model_name != saved_model:
                logger.info("Resolved embedding model: %s -> %s", saved_model, self.embedding_model_name)
        self._keyword_terms = config["keyword_terms"]
        self._hazard_class_names = config["hazard_class_names"]
        self._fitted = True
        logger.info("Feature config loaded from %s", path)
