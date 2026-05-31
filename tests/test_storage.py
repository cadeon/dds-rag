"""Tests for dds_rag.storage — Storage class."""
import os
import sys
import tempfile
import uuid
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.storage import Storage
from dds_rag.models import Card, Document, Chunk, DDCClassification


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


def _make_chunk(text, offset=0):
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        offset=offset,
        embedding=[0.1] * 384,
    )


class TestStorageInit:
    def test_creates_tables(self, tmp_db):
        storage, _ = tmp_db
        conn = storage._conn
        try:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t["name"] for t in tables]
            assert "cards" in table_names
            assert "documents" in table_names
            assert "chunks" in table_names
        finally:
            conn.close()


class TestCardCRUD:
    def test_save_and_get_card(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(ddc=530.12)
        storage.save_card(card)
        fetched = storage.get_card(card.id)
        assert fetched is not None
        assert fetched.abstract == "Test abstract"
        assert fetched.ddc_parent == 530.12

    def test_list_cards_empty(self, tmp_db):
        storage, _ = tmp_db
        assert storage.list_cards() == []

    def test_list_cards_limit(self, tmp_db):
        storage, _ = tmp_db
        for i in range(5):
            storage.save_card(_make_card(abstract=f"Card {i}"))
        cards = storage.list_cards(limit=3)
        assert len(cards) == 3

    def test_list_cards_offset(self, tmp_db):
        storage, _ = tmp_db
        for i in range(5):
            storage.save_card(_make_card(abstract=f"Card {i}"))
        cards = storage.list_cards(limit=3, offset=2)
        assert len(cards) == 3

    def test_delete_card(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        result = storage.delete_card(card.id)
        assert result is True
        assert storage.get_card(card.id) is None

    def test_delete_nonexistent(self, tmp_db):
        storage, _ = tmp_db
        result = storage.delete_card("nonexistent")
        assert result is False

    def test_get_nonexistent(self, tmp_db):
        storage, _ = tmp_db
        assert storage.get_card("nonexistent") is None

    def test_update_card(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(abstract="Old")
        storage.save_card(card)
        card.abstract = "New"
        storage.update_card(card)
        fetched = storage.get_card(card.id)
        assert fetched.abstract == "New"


class TestDocumentCRUD:
    def test_save_and_get_document(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        chunk = _make_chunk("Document content here")
        doc = Document(
            id=str(uuid.uuid4()),
            card_id=card.id,
            source="test",
            chunks=[chunk],
        )
        storage.save_document(doc)
        fetched = storage.get_document(doc.id)
        assert fetched is not None
        assert len(fetched.chunks) == 1
        assert "Document content" in fetched.chunks[0].text

    def test_get_document_by_card(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        chunk = _make_chunk("Content")
        doc = Document(id=str(uuid.uuid4()), card_id=card.id, source="test", chunks=[chunk])
        storage.save_document(doc)
        fetched = storage.get_document_by_card(card.id)
        assert fetched is not None

    def test_get_document_no_card(self, tmp_db):
        storage, _ = tmp_db
        assert storage.get_document("nope") is None


class TestSearch:
    def test_bm25_search(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(abstract="Python is great for testing")
        storage.save_card(card)
        results = storage.bm25_search("Python testing")
        assert len(results) >= 1
        assert results[0][0] == card.id

    def test_bm25_search_no_results(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(abstract="Physics and quantum mechanics")
        storage.save_card(card)
        results = storage.bm25_search("cooking recipes")
        assert len(results) == 0

    def test_bm25_candidate_ids(self, tmp_db):
        storage, _ = tmp_db
        card1 = _make_card(abstract="Python programming")
        card2 = _make_card(abstract="Rust programming")
        storage.save_card(card1)
        storage.save_card(card2)
        results = storage.bm25_search("programming", [card1.id])
        assert len(results) == 1
        assert results[0][0] == card1.id

    def test_vector_search(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        card.embedding = [1.0] + [0.0] * 383
        storage.save_card(card)
        results = storage.vector_search([1.0] + [0.0] * 383)
        assert len(results) >= 1
        assert results[0][0] == card.id
        assert results[0][1] > 0.99

    def test_vector_search_zero_query(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        results = storage.vector_search([0.0] * 384)
        assert results == []

    def test_vector_search_candidate_ids(self, tmp_db):
        storage, _ = tmp_db
        card1 = _make_card()
        card1.embedding = [1.0] + [0.0] * 383
        card2 = _make_card()
        card2.embedding = [0.0] + [1.0] * 383
        storage.save_card(card1)
        storage.save_card(card2)
        results = storage.vector_search([1.0] + [0.0] * 383, [card2.id])
        assert len(results) == 1
        assert results[0][0] == card2.id

    def test_pull_chunks(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        chunk = _make_chunk("Chunk text")
        doc = Document(id=str(uuid.uuid4()), card_id=card.id, source="test", chunks=[chunk])
        storage.save_document(doc)
        chunks = storage.pull_chunks([card.id])
        assert len(chunks) == 1
        assert chunks[0].text == "Chunk text"

    def test_pull_chunks_empty(self, tmp_db):
        storage, _ = tmp_db
        chunks = storage.pull_chunks([])
        assert chunks == []


class TestDDC:
    def test_cards_by_ddc(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(ddc=530.12)
        storage.save_card(card)
        results = storage.cards_by_ddc(530.12)
        assert len(results) == 1

    def test_cards_by_ddc_child(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(ddc=530.12)
        storage.save_card(card)
        results = storage.cards_by_ddc(530)
        assert len(results) == 1

    def test_cards_by_ddc_parent(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(ddc=530.12)
        storage.save_card(card)
        results = storage.cards_by_ddc_parent(530.12)
        assert len(results) == 1

    def test_cards_by_tag(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card(tags=["python", "testing"])
        storage.save_card(card)
        results = storage.cards_by_tag("python")
        assert len(results) == 1

    def test_is_child_main_class(self, tmp_db):
        storage, _ = tmp_db
        assert storage._is_child(530.12, 500) is True
        assert storage._is_child(599.9, 500) is True
        assert storage._is_child(600.0, 500) is False

    def test_is_child_10_level(self, tmp_db):
        storage, _ = tmp_db
        assert storage._is_child(516.37, 510) is True
        assert storage._is_child(519.9, 510) is True
        assert storage._is_child(520.0, 510) is False

    def test_is_child_exact(self, tmp_db):
        storage, _ = tmp_db
        assert storage._is_child(530.12, 530.12) is True


class TestDeleteCascades:
    def test_delete_card_removes_document_and_chunks(self, tmp_db):
        storage, _ = tmp_db
        card = _make_card()
        storage.save_card(card)
        chunk = _make_chunk("Chunk")
        doc = Document(id=str(uuid.uuid4()), card_id=card.id, source="test", chunks=[chunk])
        storage.save_document(doc)
        storage.delete_card(card.id)
        conn = storage._conn
        try:
            docs = conn.execute("SELECT COUNT(*) FROM documents WHERE card_id=?", (card.id,)).fetchone()[0]
            assert docs == 0
        finally:
            conn.close()


class TestStats:
    def test_total_cards(self, tmp_db):
        storage, _ = tmp_db
        assert storage.total_cards() == 0
        storage.save_card(_make_card())
        assert storage.total_cards() == 1

    def test_total_documents(self, tmp_db):
        storage, _ = tmp_db
        assert storage.total_documents() == 0
        card = _make_card()
        storage.save_card(card)
        doc = Document(id=str(uuid.uuid4()), card_id=card.id, source="test", chunks=[_make_chunk("x")])
        storage.save_document(doc)
        assert storage.total_documents() == 1

    def test_ddc_distribution(self, tmp_db):
        storage, _ = tmp_db
        storage.save_card(_make_card(ddc=500.0))
        storage.save_card(_make_card(ddc=500.0))
        storage.save_card(_make_card(ddc=600.0))
        dist = storage.ddc_distribution()
        assert dist["500.0"] == 2
        assert dist["600.0"] == 1
