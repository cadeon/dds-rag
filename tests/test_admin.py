"""Tests for dds_rag.admin — Admin operations."""
import os
import sys
import tempfile
import uuid
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.storage import Storage
from dds_rag.models import Card, Document, Chunk, DDCClassification
from dds_rag.admin import (
    catalog_stats, reclassify, get_card, get_document,
    search_cards, list_cards_by_ddc,
    update_card_abstract, update_card_tags, update_card_topics,
    delete_card,
)


@pytest.fixture
def tmp_db():
    path = tempfile.mktemp(suffix=".db")
    storage = Storage(path)
    yield storage, path
    if os.path.exists(path):
        os.remove(path)


def _make_card(ddc=500.0, abstract="Test abstract", tags=None, topics=None):
    return Card(
        id=str(uuid.uuid4()),
        ddc_classifications=[DDCClassification(number=ddc, confidence=0.9)],
        ddc_parent=ddc,
        abstract=abstract,
        tags=tags or ["test"],
        topics=topics or ["topic"],
        audience="general",
        format="text",
        date="2026-01-01",
        embedding=[0.1] * 384,
    )


class TestCatalogStats:
    def test_empty_catalog(self, tmp_db):
        storage, _ = tmp_db
        stats = catalog_stats(storage)
        assert stats["total_cards"] == 0
        assert stats["total_documents"] == 0
        assert stats["ddc_distribution"] == {}

    def test_populated_catalog(self, tmp_db):
        storage, _ = tmp_db
        storage.save_card(_make_card(ddc=500.0))
        storage.save_card(_make_card(ddc=600.0))
        stats = catalog_stats(storage)
        assert stats["total_cards"] == 2


class TestGetCard:
    def test_existing_card(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        result = get_card(card.id, storage)
        assert result is not None
        assert result.id == card.id

    def test_nonexistent_card(self, tmp_db):
        storage, _ = tmp_db
        result = get_card("nonexistent", storage)
        assert result is None


class TestGetDocument:
    def test_existing_document(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        chunk = Chunk(id=str(uuid.uuid4()), text="Content", offset=0, embedding=[0.1]*384)
        doc = Document(id=str(uuid.uuid4()), card_id=card.id, source="test", chunks=[chunk])
        storage.save_document(doc)
        result = get_document(card.id, storage)
        assert result is not None


class TestSearchCards:
    def test_search_finds_card(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(abstract="Python programming language")
        storage.save_card(card)
        results = search_cards("Python programming", storage)
        assert len(results) >= 1

    def test_search_no_results(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(abstract="Quantum physics")
        storage.save_card(card)
        results = search_cards("cooking recipes", storage)
        assert len(results) == 0


class TestListCardsByDDC:
    def test_list_by_ddc(self, tmp_db):
        storage, _ = tmp_db
        storage.save_card(_make_card(ddc=530.12))
        storage.save_card(_make_card(ddc=600.0))
        results = list_cards_by_ddc(530, storage)
        assert len(results) == 1


class TestUpdateCardAbstract:
    def test_update_abstract(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(abstract="Old")
        storage.save_card(card)
        result = update_card_abstract(card.id, "New abstract", storage)
        assert result is not None
        assert result.abstract == "New abstract"

    def test_update_nonexistent(self, tmp_db):
        storage, _ = tmp_db
        result = update_card_abstract("nope", "text", storage)
        assert result is None


class TestUpdateCardTags:
    def test_update_tags(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        result = update_card_tags(card.id, ["new", "tags"], storage)
        assert result is not None
        assert "new" in result.tags
        assert "tags" in result.tags


class TestUpdateCardTopics:
    def test_update_topics(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        result = update_card_topics(card.id, ["topic1"], storage)
        assert result is not None
        assert "topic1" in result.topics


class TestReclassify:
    def test_reclassify(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(ddc=500.0)
        storage.save_card(card)
        result = reclassify(card.id, 600.0, storage)
        assert result is not None
        assert result.ddc_classifications[0].number == 600.0

    def test_reclassify_nonexistent(self, tmp_db):
        storage, _ = tmp_db
        result = reclassify("nope", 600.0, storage)
        assert result is None


class TestDeleteCard:
    def test_delete(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        result = delete_card(card.id, storage)
        assert result is True
        assert get_card(card.id, storage) is None
