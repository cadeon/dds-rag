"""Card writer — LLM-powered classification for UniversalDecimalInator."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path

import requests
import yaml

from UniversalDecimalInator.models import Card, UDCClassification, Source, Artifact
from UniversalDecimalInator.reference import ClassificationReference

logger = logging.getLogger(__name__)


def _describe_image_vlm(image_path: Path) -> str:
    """Describe an image using the configured VLM endpoint.

    Reads VLM config from config.yaml under 'vlm' key:
      vlm:
        endpoint: "https://vlm.example.com/v1"
        model: "some-vision-model"

    Returns empty string if VLM is not configured or call fails.
    """
    config = _load_config()
    vlm_cfg = config.get("vlm", {})
    if not vlm_cfg.get("endpoint"):
        return ""

    import base64

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
        if api_key:
            payload["api_key"] = api_key

        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.debug("VLM description failed for %s: %s", image_path, e)
        return ""


def _download_image(url: str, dest_dir: Path, index: int) -> Path | None:
    """Download an image to the artifact directory. Returns path or None."""
    import mimetypes
    ext = mimetypes.guess_extension(
        _guess_mime_from_url(url),
    ) or ".jpg"
    filename = f"image_{index:03d}{ext}"
    dest = dest_dir / filename
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "UniversalDecimalInator/1.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        logger.debug("Failed to download image %s: %s", url, e)
        return None


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


def _make_id() -> str:
    """Generate a unique card ID as a UUID4."""
    return str(uuid.uuid4())


class CardWriter:
    """Generate classified cards from documents using LLM."""

    def __init__(self, reference: ClassificationReference):
        self.reference = reference

    def write_card(self, title: str, content: str, source_url: str = "", author: str = "", format: str = "url_fetch", kb_path: str | Path | None = None, image_info: list[dict] | None = None) -> Card:
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

        # Build artifacts list from image info
        artifacts = []
        if image_info:
            for img in image_info:
                artifacts.append(Artifact(
                    filename=img.get("filename", ""),
                    mime_type=img.get("content_type", "image/jpeg"),
                    uri=img.get("src", img.get("uri", "")),
                    description=img.get("description", ""),
                ))

        # Look up human-readable UDC label
        udc_label = self.reference.get_label(primary)

        # Generate unique ID
        final_title = result.get("title", title)
        card_id = _make_id()

        return Card(
            id=card_id,
            title=final_title,
            abstract=result.get("abstract", ""),
            classification=classification,
            tags=result.get("tags", []),
            topics=result.get("topics", []),
            sources=sources,
            artifacts=artifacts,
            source_url=source_url,
            author=author,
            content=content,
            format=format,
            udc_label=udc_label,
        )
