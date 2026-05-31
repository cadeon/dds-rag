"""Tests for filesystem operations (fs.py)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.fs import (
    card_to_markdown,
    markdown_to_card,
    write_card,
    read_card,
    list_cards,
    cards_by_classification,
    search_cards,
    catalog_stats,
)
from UniversalDecimalInator.models import Card, UDCClassification
from UniversalDecimalInator.reference import ClassificationReference

UDC_REF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "udc_reference.yaml",
)


@pytest.fixture
def ref():
    return ClassificationReference(UDC_REF)


@pytest.fixture
def kb(tmp_path):
    return str(tmp_path / "kb")


@pytest.fixture
def sample_card():
    return Card(
        id="test-ml",
        title="Machine Learning Basics",
        abstract="An introduction to machine learning concepts.",
        classification=UDCClassification(primary="004.738.5"),
        tags=["machine-learning", "ai"],
        topics=["Computer Science"],
        content="This is about machine learning.",
    )


@pytest.fixture
def sample_card2():
    return Card(
        id="test-physics",
        title="Quantum Physics",
        abstract="Introduction to quantum mechanics.",
        classification=UDCClassification(primary="530.12"),
        tags=["physics", "quantum"],
        topics=["Physics"],
        content="This is about quantum physics.",
    )


class TestCardToMarkdown:
    def test_round_trip(self, sample_card):
        md = card_to_markdown(sample_card)
        assert "---" in md
        assert "id: test-ml" in md
        assert "Machine Learning Basics" in md

    def test_classification_included(self, sample_card):
        md = card_to_markdown(sample_card)
        assert "004.738.5" in md


class TestMarkdownToCard:
    def test_parse_card(self, sample_card):
        md = card_to_markdown(sample_card)
        card = markdown_to_card(md)
        assert card.id == "test-ml"
        assert card.title == "Machine Learning Basics"
        assert card.classification.primary == "004.738.5"

    def test_plain_text(self):
        card = markdown_to_card("just some text")
        assert card.id == "unknown"
        assert card.content == "just some text"


class TestWriteCard:
    def test_creates_file(self, sample_card, kb, ref):
        path = write_card(sample_card, kb, ref)
        assert path.exists()
        assert path.suffix == ".md"

    def test_path_structure(self, sample_card, kb, ref):
        path = write_card(sample_card, kb, ref)
        parts = str(path.relative_to(kb).with_suffix("")).split("/")
        assert parts[0] == "0"  # main class
        assert parts[1] == "004"  # subdivision

    def test_creates_sources(self, sample_card, kb, ref):
        sample_card.source_url = "https://example.com/ml"
        write_card(sample_card, kb, ref)
        src = Path(kb) / "sources" / "test-ml.txt"
        assert src.exists()
        assert src.read_text() == "https://example.com/ml"


class TestReadCard:
    def test_read_existing(self, sample_card, kb, ref):
        write_card(sample_card, kb, ref)
        card = read_card(kb, "test-ml")
        assert card is not None
        assert card.title == "Machine Learning Basics"

    def test_read_nonexistent(self, kb):
        card = read_card(kb, "does-not-exist")
        assert card is None


class TestListCards:
    def test_empty_kb(self, kb):
        cards = list_cards(kb)
        assert cards == []

    def test_list_after_write(self, sample_card, sample_card2, kb, ref):
        write_card(sample_card, kb, ref)
        write_card(sample_card2, kb, ref)
        cards = list_cards(kb)
        assert len(cards) == 2


class TestCardsByClassification:
    def test_find_by_class(self, sample_card, kb, ref):
        write_card(sample_card, kb, ref)
        cards = cards_by_classification(kb, "004.738.5", ref)
        assert len(cards) == 1
        assert cards[0].id == "test-ml"

    def test_no_match(self, sample_card, kb, ref):
        write_card(sample_card, kb, ref)
        cards = cards_by_classification(kb, "530.12", ref)
        assert cards == []


class TestSearchCards:
    def test_search_by_title(self, sample_card, kb, ref):
        write_card(sample_card, kb, ref)
        results = search_cards(kb, "Machine Learning")
        assert len(results) == 1

    def test_search_by_content(self, sample_card, kb, ref):
        write_card(sample_card, kb, ref)
        results = search_cards(kb, "machine learning")
        assert len(results) == 1

    def test_search_by_tag(self, sample_card, kb, ref):
        write_card(sample_card, kb, ref)
        results = search_cards(kb, "ai")
        assert len(results) == 1

    def test_no_results(self, kb):
        results = search_cards(kb, "nonexistent")
        assert results == []


class TestCatalogStats:
    def test_empty_kb(self, kb):
        stats = catalog_stats(kb)
        assert stats["total_cards"] == 0

    def test_stats_after_write(self, sample_card, sample_card2, kb, ref):
        write_card(sample_card, kb, ref)
        write_card(sample_card2, kb, ref)
        stats = catalog_stats(kb)
        assert stats["total_cards"] == 2
        assert "0" in stats["main_class_distribution"]
        assert "5" in stats["main_class_distribution"]
        assert "machine-learning" in stats["top_tags"]
