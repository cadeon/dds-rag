"""Admin operations for the UniversalDecimalInator knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path

from UniversalDecimalInator.fs import list_cards, read_card, write_card
from UniversalDecimalInator.models import Card, UDCClassification
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


def catalog_stats(kb_path: str | Path) -> dict:
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


def reclassify(
    kb_path: str | Path,
    card_id: str,
    new_classification: str,
    ref: ClassificationReference,
) -> Card | None:
    card = read_card(kb_path, card_id)
    if not card:
        return None
    card.classification = UDCClassification(primary=new_classification)
    write_card(card, kb_path, ref)
    return card


def delete_card(kb_path: str | Path, card_id: str) -> bool:
    kb = Path(kb_path)
    for md in kb.rglob("*.md"):
        if md.name == "README.md" or "/sources/" in str(md):
            continue
        card = read_card(kb_path, md.stem)
        if card and card.id == card_id:
            md.unlink()
            src = kb / "sources" / f"{card_id}.txt"
            if src.exists():
                src.unlink()
            return True
    return False
