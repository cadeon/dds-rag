"""High-level ingest API."""

from __future__ import annotations

import uuid

from dds_rag.card_writer import CardWriter
from dds_rag.embedder import Embedder
from dds_rag.models import Card, Chunk, Document
from dds_rag.storage import Storage


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(
            id=f"{uuid.uuid4()}",
            text=chunk_text,
            offset=start,
        ))
        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def ingest(
    text: str,
    source: str = "",
    card_writer: CardWriter | None = None,
    embedder: Embedder | None = None,
    storage: Storage | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Card:
    """
    Create a catalog card from raw document text.

    Full pipeline: summarize -> classify -> embed -> store card + chunks.
    """
    # Create card via LLM
    card = card_writer.write(text)

    # Embed the card abstract
    card.embedding = embedder.embed(card.abstract)

    # Chunk and embed the document text
    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    for chunk in chunks:
        chunk.embedding = embedder.embed(chunk.text)

    # Create document
    document = Document(
        id=str(uuid.uuid4()),
        card_id=card.id,
        source=source,
        chunks=chunks,
    )

    # Store everything
    storage.save_card(card)
    storage.save_document(document)

    return card


def ingest_batch(
    documents: list[str],
    sources: list[str] | None = None,
    card_writer: CardWriter | None = None,
    embedder: Embedder | None = None,
    storage: Storage | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Card]:
    """Ingest multiple documents."""
    if sources is None:
        sources = [""] * len(documents)
    return [
        ingest(doc, src, card_writer, embedder, storage, chunk_size, chunk_overlap)
        for doc, src in zip(documents, sources)
    ]
