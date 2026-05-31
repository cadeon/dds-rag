"""SQLite storage layer with FTS5 for BM25 and vector support."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dds_rag.models import (
    Card, Chunk, Document, DDCClassification,
    deserialize_json, deserialize_vector, serialize_json, serialize_vector,
)

logger = logging.getLogger(__name__)


class Storage:
    """SQLite-backed storage for cards, documents, and chunks."""

    def __init__(self, db_path: str = "dds_rag.db"):
        self.db_path = db_path
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        conn = self._conn
        try:
            # FTS5 doesn't support IF NOT EXISTS, so drop first and recreate
            conn.execute("DROP TABLE IF EXISTS cards_fts")
            conn.execute("DROP TRIGGER IF EXISTS cards_ai")
            conn.execute("DROP TRIGGER IF EXISTS cards_au")
            conn.execute("DROP TRIGGER IF EXISTS cards_ad")
            conn.commit()

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cards (
                    id TEXT PRIMARY KEY,
                    ddc_classifications BLOB NOT NULL,
                    ddc_parent REAL NOT NULL,
                    abstract TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    audience TEXT NOT NULL,
                    format TEXT NOT NULL,
                    date TEXT NOT NULL,
                    embedding BLOB
                );

                CREATE VIRTUAL TABLE cards_fts USING fts5(
                    abstract,
                    tags,
                    topics,
                    content='cards',
                    content_rowid='rowid'
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    card_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    full_text TEXT,
                    FOREIGN KEY (card_id) REFERENCES cards(id)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    offset INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                );

                CREATE INDEX IF NOT EXISTS idx_cards_ddc_parent ON cards(ddc_parent);
                CREATE INDEX IF NOT EXISTS idx_documents_card_id ON documents(card_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

                CREATE TRIGGER cards_ai AFTER INSERT ON cards BEGIN
                    INSERT INTO cards_fts(rowid, abstract, tags, topics)
                    VALUES (new.rowid, new.abstract, new.tags, new.topics);
                END;

                CREATE TRIGGER cards_au AFTER UPDATE ON cards BEGIN
                    DELETE FROM cards_fts WHERE rowid = old.rowid;
                    INSERT INTO cards_fts(rowid, abstract, tags, topics)
                    VALUES (new.rowid, new.abstract, new.tags, new.topics);
                END;

                CREATE TRIGGER cards_ad AFTER DELETE ON cards BEGIN
                    DELETE FROM cards_fts WHERE rowid = old.rowid;
                END;
            """)
            conn.commit()
        finally:
            conn.close()

    # --- Card operations ---

    def save_card(self, card: Card) -> Card:
        conn = self._conn
        try:
            ddc_blob = serialize_json([c.to_dict() for c in card.ddc_classifications])
            tags_str = json.dumps(card.tags)
            topics_str = json.dumps(card.topics)
            emb_blob = serialize_vector(card.embedding) if card.embedding else None

            conn.execute(
                """INSERT OR REPLACE INTO cards
                   (id, ddc_classifications, ddc_parent, abstract, tags, topics,
                    audience, format, date, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (card.id, ddc_blob, card.ddc_parent, card.abstract,
                 tags_str, topics_str, card.audience, card.format,
                 card.date, emb_blob),
            )
            conn.commit()
            return card
        finally:
            conn.close()

    def get_card(self, card_id: str) -> Card | None:
        conn = self._conn
        try:
            row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
            if not row:
                return None
            return self._row_to_card(row)
        finally:
            conn.close()

    def list_cards(self, limit: int = 100, offset: int = 0) -> list[Card]:
        conn = self._conn
        try:
            rows = conn.execute(
                "SELECT * FROM cards ORDER BY date DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_card(r) for r in rows]
        finally:
            conn.close()

    def cards_by_ddc(self, ddc_number: float) -> list[Card]:
        """Find cards classified under a specific DDC number or its children."""
        conn = self._conn
        try:
            # Load all cards and filter in Python for accurate DDC matching
            all_rows = conn.execute("SELECT * FROM cards").fetchall()
            results = []
            seen = set()
            for row in all_rows:
                card = self._row_to_card(row)
                for cls in card.ddc_classifications:
                    # Match exact or child
                    if cls.number == ddc_number:
                        if card.id not in seen:
                            results.append(card)
                            seen.add(card.id)
                        break
                    # Check if this card's DDC is under the requested branch
                    if self._is_child(cls.number, ddc_number):
                        if card.id not in seen:
                            results.append(card)
                            seen.add(card.id)
                        break
            return results
        finally:
            conn.close()

    def cards_by_ddc_parent(self, parent: float) -> list[Card]:
        """Find cards whose parent DDC class matches."""
        conn = self._conn
        try:
            rows = conn.execute(
                "SELECT * FROM cards WHERE ddc_parent = ?",
                (parent,),
            ).fetchall()
            return [self._row_to_card(r) for r in rows]
        finally:
            conn.close()

    def cards_by_tag(self, tag: str) -> list[Card]:
        conn = self._conn
        try:
            all_rows = conn.execute("SELECT * FROM cards").fetchall()
            results = []
            for row in all_rows:
                card = self._row_to_card(row)
                if tag in card.tags:
                    results.append(card)
            return results
        finally:
            conn.close()

    def update_card(self, card: Card) -> Card:
        """Update an existing card."""
        return self.save_card(card)

    def delete_card(self, card_id: str) -> bool:
        conn = self._conn
        try:
            # Check if card exists first
            existing = conn.execute(
                "SELECT id FROM cards WHERE id = ?", (card_id,)
            ).fetchone()
            if not existing:
                logger.debug("Card not found for deletion: %s", card_id)
                return False
            # Delete associated document and chunks first
            doc = conn.execute(
                "SELECT id FROM documents WHERE card_id = ?", (card_id,)
            ).fetchone()
            if doc:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc["id"],))
                conn.execute("DELETE FROM documents WHERE id = ?", (doc["id"],))
            conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            # FTS5 trigger handles cards_fts cleanup via AFTER DELETE trigger
            conn.commit()
            logger.info("Card deleted: %s", card_id)
            return True
        finally:
            conn.close()

    def rebuild_bm25(self) -> int:
        """Rebuild the BM25 (FTS5) index from scratch.

        Returns the number of cards reindexed.
        """
        conn = self._conn
        try:
            conn.execute("DELETE FROM cards_fts")
            rows = conn.execute(
                "SELECT rowid, abstract, tags, topics FROM cards"
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO cards_fts(rowid, abstract, tags, topics) VALUES (?, ?, ?, ?)",
                    (row["rowid"], row["abstract"], row["tags"], row["topics"]),
                )
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    # --- Document operations ---

    def save_document(self, document: Document) -> Document:
        conn = self._conn
        try:
            conn.execute(
                """INSERT OR REPLACE INTO documents (id, card_id, source, full_text)
                   VALUES (?, ?, ?, ?)""",
                (document.id, document.card_id, document.source,
                 "\n".join(c.text for c in document.chunks)),
            )
            for chunk in document.chunks:
                emb_blob = serialize_vector(chunk.embedding) if chunk.embedding else None
                conn.execute(
                    """INSERT OR REPLACE INTO chunks (id, document_id, text, embedding, offset)
                       VALUES (?, ?, ?, ?, ?)""",
                    (chunk.id, document.id, chunk.text, emb_blob, chunk.offset),
                )
            conn.commit()
            return document
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Document | None:
        conn = self._conn
        try:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not row:
                return None
            chunks_rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY offset",
                (doc_id,),
            ).fetchall()
            chunks = [self._row_to_chunk(r) for r in chunks_rows]
            return Document(
                id=row["id"],
                card_id=row["card_id"],
                source=row["source"],
                chunks=chunks,
            )
        finally:
            conn.close()

    def get_document_by_card(self, card_id: str) -> Document | None:
        conn = self._conn
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE card_id = ?", (card_id,)
            ).fetchone()
            if not row:
                return None
            return self.get_document(row["id"])
        finally:
            conn.close()

    # --- Search operations ---

    def bm25_search(self, query: str, candidate_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """BM25 search via FTS5. Returns list of (card_id, score)."""
        conn = self._conn
        try:
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                sql = f"""
                    SELECT c.id, cards_fts.rank as score
                    FROM cards_fts
                    JOIN cards c ON c.rowid = cards_fts.rowid
                    WHERE cards_fts MATCH ? AND c.id IN ({placeholders})
                    ORDER BY rank
                """
                rows = conn.execute(sql, [query] + candidate_ids).fetchall()
            else:
                rows = conn.execute(
                    """SELECT c.id, cards_fts.rank as score
                       FROM cards_fts
                       JOIN cards c ON c.rowid = cards_fts.rowid
                       WHERE cards_fts MATCH ?
                       ORDER BY rank""",
                    (query,),
                ).fetchall()
            # FTS5 rank is negative (lower = better), invert for consistency
            return [(r["id"], -r["score"]) for r in rows]
        finally:
            conn.close()

    def vector_search(
        self,
        query_vec: list[float],
        candidate_ids: list[str] | None = None,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Cosine similarity search on card embeddings. Returns (card_id, score)."""
        conn = self._conn
        try:
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                sql = f"SELECT id, embedding FROM cards WHERE embedding IS NOT NULL AND id IN ({placeholders})"
                rows = conn.execute(sql, candidate_ids).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, embedding FROM cards WHERE embedding IS NOT NULL"
                ).fetchall()

            q = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return []

            results = []
            for row in rows:
                emb = deserialize_vector(row["embedding"])
                if not emb:
                    continue
                v = np.array(emb, dtype=np.float32)
                v_norm = np.linalg.norm(v)
                if v_norm == 0:
                    continue
                similarity = float(np.dot(q, v) / (q_norm * v_norm))
                results.append((row["id"], similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        finally:
            conn.close()

    def pull_chunks(self, card_ids: list[str]) -> list[Chunk]:
        """Pull all chunks for the given card IDs."""
        conn = self._conn
        try:
            if not card_ids:
                return []
            placeholders = ",".join("?" for _ in card_ids)
            rows = conn.execute(
                f"""SELECT c.* FROM chunks c
                   JOIN documents d ON c.document_id = d.id
                   WHERE d.card_id IN ({placeholders})
                   ORDER BY c.offset""",
                card_ids,
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]
        finally:
            conn.close()

    # --- Stats ---

    def ddc_distribution(self) -> dict[str, int]:
        """Return card count per DDC number."""
        conn = self._conn
        try:
            rows = conn.execute("SELECT ddc_classifications FROM cards").fetchall()
            dist: dict[str, int] = {}
            for row in rows:
                classifications = deserialize_json(row["ddc_classifications"])
                for cls in classifications:
                    key = str(cls["number"])
                    dist[key] = dist.get(key, 0) + 1
            return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))
        finally:
            conn.close()

    def total_cards(self) -> int:
        conn = self._conn
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM cards").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def total_documents(self) -> int:
        conn = self._conn
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # --- Helpers ---

    def _row_to_card(self, row: sqlite3.Row) -> Card:
        classifications = deserialize_json(row["ddc_classifications"])
        # tags/topics are now TEXT columns (JSON strings), not BLOBs
        tags = json.loads(row["tags"])
        topics = json.loads(row["topics"])
        embedding = deserialize_vector(row["embedding"])
        return Card(
            id=row["id"],
            ddc_classifications=[DDCClassification.from_dict(c) for c in classifications],
            ddc_parent=float(row["ddc_parent"]),
            abstract=row["abstract"],
            tags=tags,
            topics=topics,
            audience=row["audience"],
            format=row["format"],
            date=row["date"],
            embedding=embedding,
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            text=row["text"],
            embedding=deserialize_vector(row["embedding"]),
            offset=row["offset"],
        )

    def _is_child(self, number: float, parent: float) -> bool:
        """Check if number is a child of parent DDC class."""
        if number == parent:
            return True
        # Check if number falls within parent's range
        # e.g., 516.37 is child of 510 (510-519), 500 (500-599)
        if parent % 100 == 0:
            # Main class: 500 matches 500-599
            return int(number) // 100 == int(parent) // 100
        elif parent % 10 == 0:
            # 10-level: 510 matches 510-519
            return int(number) // 10 == int(parent) // 10
        else:
            # Decimal: 516.3 matches 516.30-516.39
            return str(number).startswith(str(parent))
