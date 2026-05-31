"""Card Writer - creates catalog cards from raw documents using LLM."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import requests

from dds_rag.ddc import get_ddc_tree, get_parent
from dds_rag.models import Card, DDCClassification

logger = logging.getLogger(__name__)


# Load the prompt template
_TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "card_writer.md"


def _load_prompt_template() -> str:
    """Load and build the card writer prompt from the template."""
    template = _TEMPLATE_PATH.read_text()

    # Build DDC hierarchy section
    tree = get_ddc_tree()
    ddc_section = "\n"
    for main_cls in sorted(tree.keys()):
        from dds_rag.ddc import MAIN_CLASSES
        label = MAIN_CLASSES.get(main_cls, "Unknown")
        ddc_section += f"\n### {main_cls} - {label}\n"
        for num, num_label in sorted(tree[main_cls], key=lambda x: x[0]):
            ddc_section += f"- {num}: {num_label}\n"

    return template


SYSTEM_PROMPT = """You are a document classification assistant for a digital card catalog system.
Analyze the provided document and return ONLY valid JSON with this exact structure:

{
  "abstract": "50-100 word summary of the document",
  "ddc_classifications": [{"number": 516.37, "confidence": 0.9}],
  "tags": ["hyphenated", "lowercase", "keywords"],
  "topics": ["hierarchical", "subject", "areas"],
  "audience": "general|undergraduate|graduate|academic|professional",
  "format": "article|research-paper|textbook|blog-post|report|essay|other",
  "date": "YYYY-MM-DD or unknown"
}

Rules:
- Tags must be lowercase, hyphenated, no spaces
- Assign the MOST SPECIFIC DDC number you can find
- Include multiple DDC classifications if the document covers multiple topics
- Confidence: 0.9+ for clear matches, 0.7-0.9 for reasonable, below 0.7 for uncertain
- Abstract should capture the main point, not just list topics
- Return ONLY the JSON object, no markdown, no explanation"""


class CardWriter:
    """Creates catalog cards from raw documents using an LLM."""

    def __init__(
        self,
        model: str = "qwen3.6-hermes-27b",
        endpoint: str = "http://172.30.250.101:8000/v1",
        temperature: float = 0.1,
        max_abstract_length: int = 100,
    ):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.temperature = temperature
        self.max_abstract_length = max_abstract_length

    def write(self, text: str, preferred_ddc: list[float] | None = None) -> Card:
        """Create a catalog card from document text, with retries on LLM failure."""
        prompt = self._build_prompt(text, preferred_ddc)
        max_attempts = 3
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._call_llm(prompt)
                return self._parse_result(result, text)
            except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    "Card write attempt %d/%d failed: %s", attempt, max_attempts, e
                )
                if attempt < max_attempts:
                    # Slightly increase temperature on retry to encourage different output
                    self.temperature = min(self.temperature + 0.1, 0.5)
        raise RuntimeError(
            f"Failed to create card after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _build_prompt(self, text: str, preferred_ddc: list[float] | None = None) -> str:
        """Build the full prompt for the LLM."""
        from dds_rag.ddc import DDC_CLASSES, MAIN_CLASSES

        # Build compact DDC reference
        ddc_ref = "\n".join(
            f"{num}: {label}"
            for num, label in sorted(DDC_CLASSES.items())
        )

        preferred = ""
        if preferred_ddc:
            preferred = f"\n\nPrefer reusing these existing DDC numbers if applicable: {preferred_ddc}"

        prompt = f"""{SYSTEM_PROMPT}

## DDC Classification Reference
{ddc_ref}{preferred}

## Document to Analyze
---
{text[:8000]}  # Truncate to fit context window
---

Return ONLY the JSON object:
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        url = self.endpoint + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }

        logger.debug("Calling LLM: %s (temp=%.2f)", self.model, self.temperature)
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.info("LLM response received (%d chars)", len(content))
        return content

    def _parse_result(self, llm_output: str, original_text: str) -> Card:
        """Parse LLM JSON output into a Card."""
        # Extract JSON from the response
        text = llm_output.strip()

        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"Could not find JSON in LLM output: {text[:200]}")

        json_str = text[start:end]
        data = json.loads(json_str)

        # Build DDC classifications
        classifications = []
        for cls in data.get("ddc_classifications", []):
            classifications.append(DDCClassification(
                number=float(cls["number"]),
                confidence=float(cls.get("confidence", 0.5)),
            ))

        if not classifications:
            classifications.append(DDCClassification(number=0, confidence=0.1))

        # Determine parent class
        primary = max(classifications, key=lambda c: c.confidence)
        parent = get_parent(primary.number)

        # Truncate abstract if needed
        abstract = data.get("abstract", "")
        words = abstract.split()
        if len(words) > self.max_abstract_length:
            abstract = " ".join(words[:self.max_abstract_length]) + "..."

        card = Card(
            id=str(uuid.uuid4()),
            ddc_classifications=classifications,
            ddc_parent=parent,
            abstract=abstract,
            tags=data.get("tags", []),
            topics=data.get("topics", []),
            audience=data.get("audience", "general"),
            format=data.get("format", "article"),
            date=data.get("date", "unknown"),
        )
        logger.info("Card created: %s (DDC: %s)", card.id, [c.number for c in classifications])
        return card
