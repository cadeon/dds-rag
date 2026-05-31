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
from UniversalDecimalInator.models import Card, UDCClassification, Source, CARD_FORMAT_VERSION
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

    def test_has_card_section(self, sample_card):
        md = card_to_markdown(sample_card)
        assert "## Card: Machine Learning Basics" in md

    def test_has_classification_section(self, sample_card):
        md = card_to_markdown(sample_card)
        assert "## Classification" in md
        assert "Primary: 004.738.5" in md

    def test_has_content_section(self, sample_card):
        md = card_to_markdown(sample_card)
        assert "## Content" in md

    def test_has_tags_in_classification(self, sample_card):
        md = card_to_markdown(sample_card)
        assert "Tags: machine-learning, ai" in md

    def test_has_source_section(self, sample_card, kb, ref):
        sample_card.sources = [Source(type="url", uri="https://example.com/ml", content_type="text/html")]
        sample_card.author = "Test Author"
        md = card_to_markdown(sample_card)
        assert "## Source" in md
        assert "url: https://example.com/ml" in md
        assert "Author: Test Author" in md

    def test_no_source_section_when_empty(self, sample_card):
        sample_card.sources = []
        sample_card.source_url = ""
        sample_card.author = ""
        md = card_to_markdown(sample_card)
        assert "## Source" not in md

    def test_version_in_frontmatter(self, sample_card):
        md = card_to_markdown(sample_card)
        assert f"version: '{CARD_FORMAT_VERSION}'" in md or f'version: "{CARD_FORMAT_VERSION}"' in md

    def test_format_in_frontmatter(self, sample_card):
        sample_card.format = "url_fetch"
        md = card_to_markdown(sample_card)
        assert "format: url_fetch" in md

    def test_udc_label_in_frontmatter(self, sample_card):
        sample_card.udc_label = "Computer & information sciences & AI"
        md = card_to_markdown(sample_card)
        assert "udc_label" in md
        assert "Computer & information sciences" in md

    def test_sources_in_frontmatter(self, sample_card):
        sample_card.sources = [Source(type="url", uri="https://example.com", content_type="text/html")]
        md = card_to_markdown(sample_card)
        assert "sources:" in md
        assert "uri: https://example.com" in md
        assert "content_type: text/html" in md

    def test_multiple_sources(self, sample_card):
        sample_card.sources = [
            Source(type="url", uri="https://example.com/page"),
            Source(type="file", uri="local-report.pdf", content_type="application/pdf", artifacts=["report.pdf"])
        ]
        md = card_to_markdown(sample_card)
        assert "url: https://example.com/page" in md
        assert "file: local-report.pdf" in md
        assert "artifacts: report.pdf" in md

    def test_source_url_not_in_new_frontmatter(self, sample_card):
        """New cards should not have source_url in frontmatter, only sources."""
        sample_card.sources = [Source(type="url", uri="https://example.com/ml")]
        sample_card.source_url = ""
        md = card_to_markdown(sample_card)
        assert "source_url" not in md


class TestMarkdownToCard:
    def test_parse_card(self, sample_card):
        md = card_to_markdown(sample_card)
        card = markdown_to_card(md)
        assert card.id == "test-ml"
        assert card.title == "Machine Learning Basics"
        assert card.classification.primary == "004.738.5"

    def test_content_extracted_from_section(self, sample_card):
        """New format: content extracted from ## Content section only."""
        md = card_to_markdown(sample_card)
        card = markdown_to_card(md)
        assert card.content == "This is about machine learning."
        # The structured body sections should NOT be in the content
        assert "## Classification" not in card.content
        assert "## Card:" not in card.content

    def test_old_format_backward_compat(self):
        """Old format: bare content after frontmatter still works."""
        old_md = "---\nid: old-card\ntitle: Old Format\nabstract: Old style card\nclassification:\n  primary: '000'\n  secondary: []\ntags: []\ntopics: []\nsource_url: ''\nauthor: ''\ncreated_at: '2026-01-01T00:00:00'\nupdated_at: '2026-01-01T00:00:00'\n---\n\nThis is bare content with no sections."
        card = markdown_to_card(old_md)
        assert card.id == "old-card"
        assert card.content == "This is bare content with no sections."

    def test_plain_text(self):
        card = markdown_to_card("just some text")
        assert card.id == "unknown"
        assert card.content == "just some text"

    def test_source_url_migration(self):
        """Old cards with source_url should get it migrated to sources."""
        old_md = "---\nid: old-card\ntitle: Old Format\nabstract: Old style card\nclassification:\n  primary: '000'\n  secondary: []\ntags: []\ntopics: []\nsource_url: 'https://example.com/old'\nauthor: ''\ncreated_at: '2026-01-01T00:00:00'\nupdated_at: '2026-01-01T00:00:00'\n---\n\nThis is bare content with no sections."
        card = markdown_to_card(old_md)
        assert card.id == "old-card"
        assert len(card.sources) == 1
        assert card.sources[0].uri == "https://example.com/old"
        assert card.sources[0].type == "url"

    def test_new_sources_parsed(self):
        """New cards with sources list should parse correctly."""
        new_md = "---\nid: new-card\ntitle: New Format\nabstract: New style card\nclassification:\n  primary: '004.738.5'\n  secondary: []\ntags: []\ntopics: []\nsources:\n  - type: url\n    uri: 'https://example.com/new'\n    content_type: text/html\n    artifacts:\n      - page.html\nauthor: ''\nformat: url_fetch\nudc_label: Computer & AI\nversion: '1'\ncreated_at: '2026-01-01T00:00:00'\nupdated_at: '2026-01-01T00:00:00'\n---\n\n## Card: New Format\n\nNew style card\n\n## Classification\nPrimary: 004.738.5\n\n## Content\n\nThis is the actual content.\n"
        card = markdown_to_card(new_md)
        assert card.id == "new-card"
        assert len(card.sources) == 1
        assert card.sources[0].type == "url"
        assert card.sources[0].uri == "https://example.com/new"
        assert card.sources[0].content_type == "text/html"
        assert card.sources[0].artifacts == ["page.html"]
        assert card.format == "url_fetch"
        assert card.udc_label == "Computer & AI"
        assert card.version == "1"
        assert card.content == "This is the actual content."


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

    def test_creates_artifact_dir(self, sample_card, kb, ref):
        sample_card.sources = [Source(type="url", uri="https://example.com/ml")]
        write_card(sample_card, kb, ref)
        artifact_dir = Path(kb) / "artifacts" / "test-ml"
        assert artifact_dir.is_dir()


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
