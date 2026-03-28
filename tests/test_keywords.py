"""Tests for KeywordClassifier (src.keywords).

Uses the real YAML config at configs/hazmat_keywords.yaml.
No heavy models involved -- keyword matching is pure Python.
"""

import pytest

from src.compat import get_project_root
from src.keywords import KeywordClassifier

CONFIG_PATH = get_project_root() / "configs" / "hazmat_keywords.yaml"


@pytest.fixture
def classifier() -> KeywordClassifier:
    return KeywordClassifier(config_path=CONFIG_PATH)


# -- Known hazmat items should return is_hazmat=True ----------------------


class TestHazmatDetection:
    def test_gasolina_is_hazmat(self, classifier):
        result = classifier.classify("Gasolina 5L para motor")
        assert result.is_hazmat is True
        assert result.source_layer == "keyword"

    def test_bateria_litio_18650_is_hazmat(self, classifier):
        result = classifier.classify("Bateria litio 18650 recarregavel")
        assert result.is_hazmat is True

    def test_thinner_is_hazmat(self, classifier):
        result = classifier.classify("Thinner 1 litro para pintura")
        assert result.is_hazmat is True

    def test_acetona_is_hazmat(self, classifier):
        result = classifier.classify("Acetona pura 500ml")
        assert result.is_hazmat is True

    def test_hazmat_in_description(self, classifier):
        result = classifier.classify(
            "Produto quimico industrial",
            description="contém solvente e thinner",
        )
        assert result.is_hazmat is True


# -- Confidence for keyword matches should be 0.95 -----------------------


class TestConfidence:
    def test_confidence_for_hazmat_match(self, classifier):
        result = classifier.classify("Gasolina 5L")
        assert result.confidence == 0.95

    def test_confidence_for_another_match(self, classifier):
        result = classifier.classify("Bateria litio 18650")
        assert result.confidence == 0.95


# -- Exclusions should return is_hazmat=None (unresolved) -----------------


class TestExclusions:
    def test_acido_hialuronico_excluded(self, classifier):
        """Cosmetic acid in title triggers exclusion, returns None."""
        result = classifier.classify("\u00e1cido hialur\u00f4nico creme facial")
        assert result.is_hazmat is None
        assert result.confidence == 0.0

    def test_camiseta_excluded(self, classifier):
        result = classifier.classify("Camiseta estampada rock")
        assert result.is_hazmat is None

    def test_livro_excluded(self, classifier):
        result = classifier.classify("Livro de quimica avancada")
        assert result.is_hazmat is None


# -- Non-matching items should return is_hazmat=None ----------------------


class TestNonMatching:
    def test_camiseta_algodao(self, classifier):
        result = classifier.classify("camiseta algodao masculina")
        assert result.is_hazmat is None
        assert result.confidence == 0.0

    def test_generic_product(self, classifier):
        result = classifier.classify("Mesa de escritorio com gaveta")
        assert result.is_hazmat is None

    def test_empty_title(self, classifier):
        result = classifier.classify("")
        assert result.is_hazmat is None

    def test_source_layer_always_keyword(self, classifier):
        result = classifier.classify("qualquer produto aleatorio")
        assert result.source_layer == "keyword"
