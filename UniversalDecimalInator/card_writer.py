"""Card writer — LLM-powered classification for UniversalDecimalInator."""

from __future__ import annotations

import json
import logging
import re

from UniversalDecimalInator.models import Card, UDCClassification
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a classification expert. Given a document, classify it using Universal Decimal Classification (UDC).

{reference}

RULES:
- Return ONLY valid JSON, no markdown, no explanation
- primary_classification: single UDC number (may be compound with ':')
- secondary_classifications: list of additional UDC numbers
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
  "primary_classification": "...",
  "secondary_classifications": ["...", "..."],
  "tags": ["...", "..."],
  "topics": ["...", "..."]
}}"""


def _classify_with_llm(title: str, content: str, ref: ClassificationReference) -> dict:
    import urllib.request
    import urllib.error

    prompt = USER_PROMPT.format(title=title[:500], content=content[:3000])
    system = SYSTEM_PROMPT.format(reference=ref.to_prompt_reference()[:6000])

    payload = json.dumps({
        "model": "qwen3.6-hermes-27b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }).encode()

    req = urllib.request.Request(
        "http://172.30.250.101:8000/v1/chat/completions",
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


class CardWriter:
    """Generate classified cards from documents using LLM."""

    def __init__(self, reference: ClassificationReference):
        self.reference = reference

    def write_card(self, title: str, content: str, source_url: str = "", author: str = "") -> Card:
        result = _classify_with_llm(title, content, self.reference)

        primary = result.get("primary_classification", "000")
        if not self.reference.validate(primary):
            primary = "000"

        classification = UDCClassification(
            primary=primary,
            secondary=result.get("secondary_classifications", []),
        )

        return Card(
            id=title.lower().replace(" ", "-")[:50],
            title=result.get("title", title),
            abstract=result.get("abstract", ""),
            classification=classification,
            tags=result.get("tags", []),
            topics=result.get("topics", []),
            source_url=source_url,
            author=author,
            content=content,
        )
