"""High-level query API."""

from __future__ import annotations

from dataclasses import dataclass

from dds_rag.embedder import Embedder
from dds_rag.models import Chunk
from dds_rag.query_router import QueryRouter
from dds_rag.reranker import Reranker
from dds_rag.storage import Storage


@dataclass
class Result:
    """A query result with chunk text and metadata."""
    chunk: Chunk
    card_id: str
    score: float
    source: str = ""


def query(
    text: str,
    top_k: int = 10,
    embedder: Embedder | None = None,
    query_router: QueryRouter | None = None,
    reranker: Reranker | None = None,
    storage: Storage | None = None,
    fused_top_k: int = 20,
) -> list[Result]:
    """
    Hybrid search the card catalog, pull documents, return ranked chunks.

    Pipeline: query -> DDC routing -> hybrid search -> RRF -> pull chunks -> rerank
    """
    # Embed the query
    query_vec = embedder.embed(text)

    # Route through the catalog
    chunks = query_router.route(
        query=text,
        query_vec=query_vec,
        storage=storage,
        reranker=reranker,
        top_k=top_k,
        fused_top_k=fused_top_k,
    )

    # Build results with metadata
    results = []
    seen_cards = set()
    for chunk in chunks:
        # Find which card this chunk belongs to
        doc = storage.get_document(chunk.id.replace(chunk.id.split("-")[-1], ""))
        # Actually, we need to trace back from chunk to document to card
        # Let's do a simpler approach
        results.append(Result(
            chunk=chunk,
            card_id="",  # Will be filled in below
            score=0.0,
        ))

    # Map chunks back to cards via documents
    chunk_to_card: dict[str, str] = {}
    chunk_to_source: dict[str, str] = {}
    for card in storage.list_cards(limit=10000):
        doc = storage.get_document_by_card(card.id)
        if doc:
            for chunk in doc.chunks:
                chunk_to_card[chunk.id] = card.id
                chunk_to_source[chunk.id] = doc.source

    for result in results:
        result.card_id = chunk_to_card.get(result.chunk.id, "")
        result.source = chunk_to_source.get(result.chunk.id, "")

    return results[:top_k]


def query_ddc(
    text: str,
    ddc_number: float,
    top_k: int = 10,
    embedder: Embedder | None = None,
    query_router: QueryRouter | None = None,
    reranker: Reranker | None = None,
    storage: Storage | None = None,
    fused_top_k: int = 20,
) -> list[Result]:
    """Hybrid search within a specific DDC branch."""
    query_vec = embedder.embed(text)

    chunks = query_router.route_ddc(
        query=text,
        query_vec=query_vec,
        ddc_number=ddc_number,
        storage=storage,
        reranker=reranker,
        top_k=top_k,
        fused_top_k=fused_top_k,
    )

    results = []
    for chunk in chunks:
        results.append(Result(
            chunk=chunk,
            card_id="",
            score=0.0,
        ))

    return results[:top_k]
