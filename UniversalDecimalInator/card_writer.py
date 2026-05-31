"""Card writer — LLM-powered classification for UniversalDecimalInator."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import yaml

from UniversalDecimalInator.models import Card, UDCClassification, Source
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a classification expert. Given a document, classify it using Universal Decimal Classification (UDC).

{reference}

RULES:
- Return ONLY valid JSON, no markdown, no explanation
- classification.primary: single UDC number (may be compound with ':')
- classification.secondary: list of additional UDC numbers
- tags: 3-8 descriptive keywords
- topics: 1-3 high-level topic areas
- abstract: 2-4 sentence summary
- title: concise descriptive title"""

USER_PROMPT = """Classify this document:

Title: {title}
Content: {content}

Respond with JSON:
{{
  "title": "...",
  "abstract": "...",
  "classification": {{
    "primary": "...",
    "secondary": ["...", "..."]
  }},
  "tags": ["...", "..."],
  "topics": ["...", "..."]
}}"""


def _load_config() -> dict:
    """Load card_writer config from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _classify_with_llm(title: str, content: str, ref: ClassificationReference) -> dict:
    import urllib.request
    import urllib.error

    config = _load_config()
    writer_cfg = config.get("card_writer", {})
    endpoint = writer_cfg.get("endpoint", "http://172.30.250.101:8000/v1")
    model = writer_cfg.get("model", "qwen3.6-hermes-27b")
    temperature = writer_cfg.get("temperature", 0.1)

    prompt = USER_PROMPT.format(title=title[:500], content=content[:3000])
    system = SYSTEM_PROMPT.format(reference=ref.to_prompt_reference()[:6000])

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1000,
    }).encode()

    url = f"{endpoint}/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM call failed: {e}")

    response = data["choices"][0]["message"]["content"]

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON in LLM response: {response[:200]}")

    return json.loads(json_match.group())


def _make_id(title: str, kb_path: str | Path | None = None) -> str:
    """Generate a unique card ID from a title.

    Uses a slugified title with a short hash suffix to avoid collisions.
    If kb_path is provided, checks for existing IDs and appends a counter
    if the generated ID already exists.
    """
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower().strip()).strip('-')[:40]
    # Add short hash for uniqueness
    short_hash = hashlib.sha256((title + str(time.time())).encode()).hexdigest()[:6]
    candidate = f"{slug}-{short_hash}"

    if kb_path:
        # Check for collisions
        kb = Path(kb_path)
        existing = set()
        for md in kb.rglob("*.md"):
            if md.name == "README.md" or "/sources/" in str(md) or "/artifacts/" in str(md):
                continue
            try:
                fm_text = md.read_text().split("---", 2)[1] if md.read_text().startswith("---") else ""
                fm = yaml.safe_load(fm_text)
                if fm and fm.get("id"):
                    existing.add(fm["id"])
            except Exception:
                pass
        if candidate not in existing:
            return candidate
        # Fallback: keep appending counter
        counter = 1
        while f"{candidate}-{counter}" in existing:
            counter += 1
        return f"{candidate}-{counter}"

    return candidate


class CardWriter:
    """Generate classified cards from documents using LLM."""

    def __init__(self, reference: ClassificationReference):
        self.reference = reference

    def write_card(self, title: str, content: str, source_url: str = "", author: str = "", format: str = "url_fetch", kb_path: str | Path | None = None) -> Card:
        result = _classify_with_llm(title, content, self.reference)

        cls_data = result.get("classification", {})
        primary = cls_data.get("primary", "000")
        if not self.reference.validate(primary):
            primary = "000"

        classification = UDCClassification(
            primary=primary,
            secondary=cls_data.get("secondary", []),
        )

        # Build sources list
        sources = []
        if source_url:
            sources.append(Source(type="url", uri=source_url))

        # Look up human-readable UDC label
        udc_label = self.reference.get_label(primary)

        # Generate unique ID from the LLM-provided title
        final_title = result.get("title", title)
        card_id = _make_id(final_title, kb_path)

        return Card(
            id=card_id,
            title=final_title,
            abstract=result.get("abstract", ""),
            classification=classification,
            tags=result.get("tags", []),
            topics=result.get("topics", []),
            sources=sources,
            source_url=source_url,
            author=author,
            content=content,
            format=format,
            udc_label=udc_label,
        )
