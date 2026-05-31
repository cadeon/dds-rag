"""High-level ingest API for UniversalDecimalInator."""

from __future__ import annotations

import logging
from pathlib import Path

from UniversalDecimalInator.card_writer import CardWriter
from UniversalDecimalInator.fetcher import fetch_url
from UniversalDecimalInator.fs import write_card
from UniversalDecimalInator.models import Card
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


def ingest(
    title: str,
    content: str,
    kb_path: str | Path,
    ref: ClassificationReference,
    source_url: str = "",
    author: str = "",
) -> Card:
    writer = CardWriter(reference=ref)
    card = writer.write_card(title, content, source_url, author, kb_path=kb_path)
    write_card(card, kb_path, ref)
    logger.info("Ingested: %s -> %s", title, card.classification.primary)
    return card


def ingest_url(
    url: str,
    kb_path: str | Path,
    ref: ClassificationReference,
    author: str = "",
) -> Card:
    content = fetch_url(url)
    title = url.split("/")[-1] or url
    return ingest(title, content, kb_path, ref, source_url=url, author=author)


def ingest_batch(
    items: list[dict],
    kb_path: str | Path,
    ref: ClassificationReference,
) -> list[Card]:
    cards = []
    for item in items:
        try:
            card = ingest(
                title=item["title"],
                content=item["content"],
                kb_path=kb_path,
                ref=ref,
                source_url=item.get("source_url", ""),
                author=item.get("author", ""),
            )
            cards.append(card)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", item.get("title", "unknown"), e)
    return cards
