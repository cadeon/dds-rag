"""Filesystem operations for UniversalDecimalInator knowledge base."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from UniversalDecimalInator.models import Card, UDCClassification, Source, CARD_FORMAT_VERSION
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


def card_to_markdown(card: Card) -> str:
    """Serialize a Card to RAG-friendly markdown.

    Output structure:
      - YAML frontmatter with full metadata (machine-readable)
      - ## Card section: title + abstract (summary chunk for RAG)
      - ## Classification section: UDC + tags + topics (metadata travels with chunks)
      - ## Content section: original document text (what gets chunked)
      - ## Source section: provenance (URL, author)

    This format ensures RAG chunkers that respect markdown headings get
    metadata context in every chunk, while the frontmatter provides
    programmatic access to the complete card.
    """
    frontmatter = {
        "id": card.id,
        "title": card.title,
        "abstract": card.abstract,
        "classification": card.classification.to_dict(),
        "tags": card.tags,
        "topics": card.topics,
        "author": card.author,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    }
    if card.sources:
        frontmatter["sources"] = [s.to_dict() for s in card.sources]
    if card.format:
        frontmatter["format"] = card.format
    if card.udc_label:
        frontmatter["udc_label"] = card.udc_label
    frontmatter["version"] = card.version or CARD_FORMAT_VERSION
    fm = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

    # Build structured body sections
    body_parts = []

    # Card summary section
    card_header = f"## Card: {card.title}" if card.title else f"## Card: {card.id}"
    body_parts.append(card_header)
    if card.abstract:
        body_parts.append(card.abstract)

    # Classification section — metadata repeated for RAG chunk context
    cls_lines = [f"Primary: {card.classification.primary}"]
    if card.classification.secondary:
        cls_lines.append(f"Secondary: {', '.join(card.classification.secondary)}")
    if card.tags:
        cls_lines.append(f"Tags: {', '.join(card.tags)}")
    if card.topics:
        cls_lines.append(f"Topics: {', '.join(card.topics)}")
    body_parts.append("## Classification")
    body_parts.append(" | ".join(cls_lines))

    # Content section — original document text
    body_parts.append("## Content")
    body_parts.append(card.content)

    # Source section — provenance
    has_sources = card.sources or card.source_url or card.author
    if has_sources:
        src_lines = ["## Source"]
        for src in card.sources:
            if src.uri:
                src_lines.append(f"- {src.type}: {src.uri}")
                if src.content_type:
                    src_lines.append(f"  content_type: {src.content_type}")
        if card.author:
            src_lines.append(f"Author: {card.author}")
        body_parts.append("\n".join(src_lines))

    return f"---\n{fm}---\n\n" + "\n\n".join(body_parts)


def markdown_to_card(text: str) -> Card:
    """Parse a markdown file back into a Card.

    Handles both formats:
    - New format: YAML frontmatter + structured body (## Card, ## Classification,
      ## Content, ## Source sections). Content is extracted from the ## Content
      section only.
    - Old format: YAML frontmatter + bare content after the closing ---.
    - Plain text: no frontmatter at all.
    """
    if not text.startswith("---"):
        return Card(id="unknown", content=text)
    parts = text.split("---", 2)
    if len(parts) < 3:
        return Card(id="unknown", content=text)
    fm = yaml.safe_load(parts[1])
    body = parts[2].strip()

    # Try to extract content from ## Content section (new format)
    content_match = re.search(r"## Content\n(.*?)(?=\n## |\Z)", body, re.DOTALL)
    if content_match:
        content = content_match.group(1).strip()
    else:
        # Old format: entire body is content
        content = body

    fm["content"] = content
    return Card.from_dict(fm)


def write_card(card: Card, kb_path: str | Path, ref: ClassificationReference) -> Path:
    """Write a card to the knowledge base.

    Layout:
      kb/<classification_path>/<card_id>.md           -- card file
      kb/artifacts/<classification_path>/<card_id>/   -- artifact files for this card

    Card files are named by UID, not classification, so multiple cards
    can share the same classification. Artifacts mirror the classification
    directory structure with a per-card subdirectory.
    """
    kb = Path(kb_path)
    cls_path = ref.udc_to_path(card.classification.primary)

    # Card file: <kb>/<classification_path>/<card_id>.md
    card_dir = kb / cls_path
    card_dir.mkdir(parents=True, exist_ok=True)
    card_path = card_dir / f"{card.id}.md"
    card_path.write_text(card_to_markdown(card))

    # Artifact directory: <kb>/artifacts/<classification_path>/<card_id>/
    artifact_dir = kb / "artifacts" / cls_path / card.id
    artifact_dir.mkdir(parents=True, exist_ok=True)

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
        if md.name == "README.md" or "/sources/" in str(md) or "/artifacts/" in str(md):
            continue
        text = md.read_text()
        try:
            cards.append(markdown_to_card(text))
        except Exception:
            continue
    return cards


def cards_by_classification(kb_path: str | Path, classification: str, ref: ClassificationReference) -> list[Card]:
    """Find cards under a classification using directory walk + prefix matching.

    Walks the kb tree looking for card files whose directory path starts with
    the classification number (UDC is hierarchical by numeric prefix).
    Also checks secondary classifications.
    """
    kb = Path(kb_path)
    result = []
    seen = set()

    for md in kb.rglob("*.md"):
        if md.name == "README.md" or "/sources/" in str(md) or "/artifacts/" in str(md):
            continue
        # The leaf directory above the .md file is the classification
        leaf_cls = md.parent.name
        if leaf_cls.startswith(classification):
            try:
                text = md.read_text()
                card = markdown_to_card(text)
                if card.id not in seen:
                    result.append(card)
                    seen.add(card.id)
            except Exception:
                continue

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
    scored.sort(key=lambda x: (-x[0], x[1].id), reverse=False)
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
