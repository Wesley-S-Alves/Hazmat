"""Layer 1 — Keyword-based hazmat classification.

Uses deterministic keyword matching against a curated list of hazmat terms.
Resolves ~60% of items with high confidence at zero cost.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("hazmat.keywords")


@dataclass
class ClassificationResult:
    """Result of a classification attempt."""

    is_hazmat: bool | None  # None = ambiguous/unresolved
    reason: str
    confidence: float
    source_layer: str


def load_keywords(config_path: Path | None = None) -> dict:
    """Load hazmat keywords from YAML config."""
    if config_path is None:
        from src.compat import get_project_root

        config_path = get_project_root() / "configs" / "hazmat_keywords.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class KeywordClassifier:
    """Classify items as hazmat using keyword matching."""

    def __init__(self, config_path: Path | None = None):
        config = load_keywords(config_path)
        self.hazard_classes = config.get("hazard_classes", {})
        self.exclusions = [e.lower() for e in config.get("exclusions", [])]

        # Pre-compile regex patterns for each class
        self._patterns: dict[str, tuple[re.Pattern, str]] = {}
        for class_name, class_data in self.hazard_classes.items():
            keywords = class_data.get("keywords", [])
            if keywords:
                # Build alternation pattern, escape special chars
                escaped = [re.escape(kw.lower()) for kw in keywords]
                pattern = re.compile("|".join(escaped), re.IGNORECASE)
                self._patterns[class_name] = (
                    pattern,
                    class_data.get("reason", "Hazmat keyword match"),
                )

        # Pre-compile exclusion pattern
        if self.exclusions:
            escaped_exc = [re.escape(e) for e in self.exclusions]
            self._exclusion_pattern = re.compile("|".join(escaped_exc), re.IGNORECASE)
        else:
            self._exclusion_pattern = None

    def classify(self, title: str, description: str = "") -> ClassificationResult:
        """Classify a single item by keyword matching.

        Returns ClassificationResult with is_hazmat=True/False/None.
        None means the keyword layer couldn't determine — pass to next layer.
        """
        text = f"{title} {description}".lower()

        # Check exclusions first — if title matches an exclusion, skip keyword matching
        if self._exclusion_pattern and self._exclusion_pattern.search(title.lower()):
            return ClassificationResult(
                is_hazmat=None,
                reason="",
                confidence=0.0,
                source_layer="keyword",
            )

        # Check each hazard class
        for class_name, (pattern, reason) in self._patterns.items():
            match = pattern.search(text)
            if match:
                matched_keyword = match.group()
                return ClassificationResult(
                    is_hazmat=True,
                    reason=f"{reason} (matched: '{matched_keyword}')",
                    confidence=0.95,
                    source_layer="keyword",
                )

        # No match — return None (ambiguous, pass to next layer)
        return ClassificationResult(
            is_hazmat=None,
            reason="",
            confidence=0.0,
            source_layer="keyword",
        )
