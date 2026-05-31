"""UniversalDecimalInator: Universal Decimal Classification-based document catalog system.

Pluggable classification reference — swap UDC for DDC, LCC, or any custom
taxonomy by providing a different reference YAML file.
"""

__version__ = "0.1.0"

from UniversalDecimalInator.reference import ClassificationReference
from UniversalDecimalInator.card_writer import CardWriter
from UniversalDecimalInator.models import Card, UDCClassification, Source, CARD_FORMAT_VERSION
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
from UniversalDecimalInator.ingest import ingest, ingest_url, ingest_batch
from UniversalDecimalInator.admin import reclassify, delete_card
from UniversalDecimalInator.fetcher import fetch_url

__all__ = [
    "ClassificationReference",
    "CardWriter",
    "Card",
    "UDCClassification",
    "Source",
    "CARD_FORMAT_VERSION",
    "card_to_markdown",
    "markdown_to_card",
    "write_card",
    "read_card",
    "list_cards",
    "cards_by_classification",
    "search_cards",
    "catalog_stats",
    "ingest",
    "ingest_url",
    "ingest_batch",
    "reclassify",
    "delete_card",
    "fetch_url",
]
