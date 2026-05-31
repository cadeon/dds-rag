"""Admin operations for the UniversalDecimalInator knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from UniversalDecimalInator.fs import list_cards, read_card, write_card
from UniversalDecimalInator.models import Card, UDCClassification
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


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
        text = md.read_text()
        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        fm = yaml.safe_load(parts[1])
        if fm.get("id") == card_id:
            md.unlink()
            src = kb / "sources" / f"{card_id}.txt"
            if src.exists():
                src.unlink()
            return True
    return False
