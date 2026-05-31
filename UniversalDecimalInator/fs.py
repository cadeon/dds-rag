"""Filesystem operations for UniversalDecimalInator knowledge base."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from UniversalDecimalInator.models import Card, UDCClassification
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


def card_to_markdown(card: Card) -> str:
    frontmatter = {
        "id": card.id,
        "title": card.title,
        "abstract": card.abstract,
        "classification": card.classification.to_dict(),
        "tags": card.tags,
        "topics": card.topics,
        "source_url": card.source_url,
        "author": card.author,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    }
    fm = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    return f"---\n{fm}---\n\n{card.content}"


def markdown_to_card(text: str) -> Card:
    if not text.startswith("---"):
        return Card(id="unknown", content=text)
    parts = text.split("---", 2)
    if len(parts) < 3:
        return Card(id="unknown", content=text)
    fm = yaml.safe_load(parts[1])
    content = parts[2].strip()
    fm["content"] = content
    return Card.from_dict(fm)


def write_card(card: Card, kb_path: str | Path, ref: ClassificationReference) -> Path:
    kb = Path(kb_path)
    path = ref.udc_to_path(card.classification.primary)
    card_path = kb / f"{path}.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card_to_markdown(card))
    if card.source_url:
        sources = kb / "sources"
        sources.mkdir(exist_ok=True)
        src_file = sources / f"{card.id}.txt"
        src_file.write_text(card.source_url)
    return card_path


def read_card(kb_path: str | Path, card_id: str) -> Card | None:
    kb = Path(kb_path)
    for md in kb.rglob("*.md"):
        if md.name == "README.md":
            continue
        text = md.read_text()
        card = markdown_to_card(text)
        if card.id == card_id:
            return card
    return None


def list_cards(kb_path: str | Path) -> list[Card]:
    kb = Path(kb_path)
    cards = []
    for md in sorted(kb.rglob("*.md")):
        if md.name == "README.md" or "/sources/" in str(md):
            continue
        text = md.read_text()
        try:
            cards.append(markdown_to_card(text))
        except Exception:
            continue
    return cards


def cards_by_classification(kb_path: str | Path, classification: str, ref: ClassificationReference) -> list[Card]:
    cards = list_cards(kb_path)
    target_base = ref.strip_facets(ref.first_component(classification))
    result = []
    for card in cards:
        card_base = ref.strip_facets(ref.first_component(card.classification.primary))
        if card_base == target_base or card.classification.primary == classification:
            result.append(card)
        if classification in card.classification.secondary:
            result.append(card)
    return result


def search_cards(kb_path: str | Path, query: str, top_k: int = 10) -> list[Card]:
    cards = list_cards(kb_path)
    query_lower = query.lower()
    scored = []
    for card in cards:
        score = 0
        if query_lower in card.title.lower():
            score += 10
        if query_lower in card.abstract.lower():
            score += 5
        if query_lower in card.content.lower():
            score += 1
        for tag in card.tags:
            if query_lower in tag.lower():
                score += 3
        for term in query_lower.split():
            if term in card.classification.primary:
                score += 2
        if score > 0:
            scored.append((score, card))
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


def catalog_stats(kb_path: str | Path) -> dict:
    kb = Path(kb_path)
    cards = list_cards(kb_path)
    total = len(cards)
    main_class_dist: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for card in cards:
        mc = card.classification.udc_main_class or "uncategorized"
        main_class_dist[mc] = main_class_dist.get(mc, 0) + 1
        for tag in card.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "total_cards": total,
        "main_class_distribution": main_class_dist,
        "top_tags": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:20]),
    }
