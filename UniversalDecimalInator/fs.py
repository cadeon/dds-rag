"""Filesystem operations for UniversalDecimalInator knowledge base."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from UniversalDecimalInator.models import Card, UDCClassification, Source, Artifact, CARD_FORMAT_VERSION
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


def _is_content_image(url: str) -> bool:
    """Check if an image URL is likely a content image rather than noise."""
    # Filter out math equation renders from Wikimedia
    if "wikimedia.org/api/rest_v1/media/math/render" in url:
        return False
    return True


def _download_image(url: str, dest_dir: Path, index: int) -> tuple[Path | None, str]:
    """Download an image to the artifact directory. Returns (path, filename) or (None, '')."""
    import mimetypes
    ext = mimetypes.guess_extension(_guess_mime_from_url(url)) or ".jpg"
    filename = f"image_{index:03d}{ext}"
    dest = dest_dir / filename
    try:
        import requests
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "UniversalDecimalInator/1.0 (+https://github.com/cadeon/dds-rag)"
        })
        resp.raise_for_status()
        content = resp.content
        # Skip very small files (likely errors or tracking pixels)
        if len(content) < 1024:
            logger.debug("Skipping tiny image %s (%d bytes)", url, len(content))
            return None, ""
        dest.write_bytes(content)
        return dest, filename
    except Exception as e:
        logger.debug("Failed to download image %s: %s", url, e)
        return None, ""


def _describe_image_vlm(image_path: Path) -> str:
    """Describe an image using the configured VLM endpoint.

    Reads VLM config from config.yaml under 'vlm' key:
      vlm:
        endpoint: "https://vlm.example.com/v1"
        model: "some-vision-model"

    Returns empty string if VLM is not configured or call fails.
    """
    import json
    import os
    import yaml
    import base64
    import requests

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.pardir, "config.yaml",
    )
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return ""

    vlm_cfg = config.get("vlm", {})
    if not vlm_cfg.get("endpoint"):
        return ""

    endpoint = vlm_cfg["endpoint"]
    model = vlm_cfg.get("model", "")
    api_key = vlm_cfg.get("api_key", "")

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Briefly describe what you see in this image in 1-2 sentences."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 100,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.debug("VLM description failed for %s: %s", image_path, e)
        return ""

def _guess_mime_from_url(url: str) -> str:
    """Guess MIME type from URL extension."""
    ext = url.rsplit(".", 1)[-1].lower().split("?")[0]
    types = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "svg": "image/svg+xml",
        "bmp": "image/bmp", "tiff": "image/tiff",
    }
    return types.get(ext, "image/jpeg")


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
    if card.artifacts:
        frontmatter["artifacts"] = [a.to_dict() for a in card.artifacts]
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

    # Artifacts section — associated files with descriptions for RAG context
    valid_artifacts = [art for art in card.artifacts if art.filename]
    if valid_artifacts:
        art_lines = ["## Artifacts"]
        for art in valid_artifacts:
            art_lines.append(f"- **{art.filename}** ({art.mime_type})")
            if art.description:
                art_lines.append(f"  {art.description}")
            if art.uri:
                art_lines.append(f"  source: {art.uri}")
        body_parts.append("\n".join(art_lines))

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
      kb/content/<classification_path>/<card_id>.md     -- card file
      kb/artifacts/<classification_path>/<card_id>/     -- artifact files for this card

    Card files are named by UID, not classification, so multiple cards
    can share the same classification. Artifacts mirror the classification
    directory structure with a per-card subdirectory.
    """
    kb = Path(kb_path)
    cls_path = ref.udc_to_path(card.classification.primary)

    # Card file: <kb>/content/<classification_path>/<card_id>.md
    card_dir = kb / "content" / cls_path
    card_dir.mkdir(parents=True, exist_ok=True)

    # Artifact directory: <kb>/artifacts/<classification_path>/<card_id>/
    artifact_dir = kb / "artifacts" / cls_path / card.id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Download images and generate VLM descriptions for artifacts
    for idx, art in enumerate(card.artifacts):
        if art.uri and not art.filename:
            img_path, filename = _download_image(art.uri, artifact_dir, idx)
            if img_path:
                art.filename = filename
                # Generate VLM description
                art.description = _describe_image_vlm(img_path)
                logger.info("Artifact %s: %s", art.filename, art.description or "(no VLM)")

    card_path = card_dir / f"{card.id}.md"
    card_path.write_text(card_to_markdown(card))

    return card_path


def read_card(kb_path: str | Path, card_id: str) -> Card | None:
    kb = Path(kb_path) / "content"
    for md in kb.rglob("*.md"):
        if md.name == "README.md":
            continue
        text = md.read_text()
        card = markdown_to_card(text)
        if card.id == card_id:
            return card
    return None


def list_cards(kb_path: str | Path) -> list[Card]:
    kb = Path(kb_path) / "content"
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
    """Find cards under a classification.

    Walks the kb tree, loading each card and checking:
    - primary classification matches (exact or prefix for hierarchical UDCs)
    - primary classification first component matches (for compound UDCs like 352.4:44-2)
    - secondary classifications match
    """
    kb = Path(kb_path) / "content"
    result = []
    seen = set()

    for md in kb.rglob("*.md"):
        if md.name == "README.md" or "/sources/" in str(md) or "/artifacts/" in str(md):
            continue
        if md.stem in seen:
            continue
        try:
            text = md.read_text()
            card = markdown_to_card(text)
        except Exception:
            continue

        primary = card.classification.primary

        # Exact match
        if primary == classification:
            result.append(card)
            seen.add(card.id)
            continue

        # Hierarchical prefix match: e.g. browsing 352 matches 352.4, 352.41, etc.
        # Also handles compound UDCs: browsing 352.4:44-2 matches cards with that exact primary
        if primary.startswith(classification + ".") or primary.startswith(classification + ":"):
            result.append(card)
            seen.add(card.id)
            continue

        # Ancestor match for single-digit or short numeric parents:
        # browsing "9" should match "914", browsing "35" should match "352"
        card_first = ref.first_component(primary)
        if card_first.startswith(classification) and len(classification) < len(card_first):
            # Make sure it's a real ancestor, not just a string coincidence
            # e.g. "9" is ancestor of "914", "35" is ancestor of "352"
            # Check that the next char after classification is a digit (not a dot/dash)
            remainder = card_first[len(classification):]
            if remainder and remainder[0].isdigit():
                result.append(card)
                seen.add(card.id)
                continue

        # Check secondary classifications
        if classification in card.classification.secondary:
            result.append(card)
            seen.add(card.id)
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
