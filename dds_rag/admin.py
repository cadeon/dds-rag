"""Admin operations for the card catalog."""

from __future__ import annotations

from dds_rag.models import Card, Document
from dds_rag.storage import Storage


def catalog_stats(storage: Storage) -> dict:
    """Return catalog statistics."""
    return {
        "total_cards": storage.total_cards(),
        "total_documents": storage.total_documents(),
        "ddc_distribution": storage.ddc_distribution(),
    }


def reclassify(card_id: str, new_ddc: float, storage: Storage) -> Card | None:
    """Manually override card classification."""
    card = storage.get_card(card_id)
    if not card:
        return None

    from dds_rag.ddc import get_parent
    from dds_rag.models import DDCClassification

    card.ddc_classifications = [DDCClassification(number=new_ddc, confidence=1.0)]
    card.ddc_parent = get_parent(new_ddc)

    storage.update_card(card)
    return card


def get_card(card_id: str, storage: Storage) -> Card | None:
    """Look up a card by ID."""
    return storage.get_card(card_id)


def get_document(card_id: str, storage: Storage) -> Document | None:
    """Pull the full document behind a card."""
    return storage.get_document_by_card(card_id)


def search_cards(
    query: str,
    storage: Storage,
    limit: int = 50,
) -> list[Card]:
    """Search cards by keyword (BM25 on abstract/tags/topics)."""
    results = storage.bm25_search(query)
    cards = []
    for card_id, score in results[:limit]:
        card = storage.get_card(card_id)
        if card:
            cards.append(card)
    return cards


def list_cards_by_ddc(
    ddc_number: float,
    storage: Storage,
) -> list[Card]:
    """List all cards in a DDC branch."""
    return storage.cards_by_ddc(ddc_number)


def update_card_abstract(
    card_id: str,
    abstract: str,
    storage: Storage,
    embedder=None,
) -> Card | None:
    """Update a card's abstract and re-embed it."""
    card = storage.get_card(card_id)
    if not card:
        return None

    card.abstract = abstract
    if embedder:
        card.embedding = embedder.embed(abstract)

    storage.update_card(card)
    return card


def update_card_tags(
    card_id: str,
    tags: list[str],
    storage: Storage,
) -> Card | None:
    """Update a card's tags."""
    card = storage.get_card(card_id)
    if not card:
        return None

    card.tags = tags
    storage.update_card(card)
    return card


def update_card_topics(
    card_id: str,
    topics: list[str],
    storage: Storage,
) -> Card | None:
    """Update a card's topics."""
    card = storage.get_card(card_id)
    if not card:
        return None

    card.topics = topics
    storage.update_card(card)
    return card


def delete_card(card_id: str, storage: Storage) -> bool:
    """Delete a card and its associated document."""
    return storage.delete_card(card_id)
