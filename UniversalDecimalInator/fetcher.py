"""URL fetching and content extraction for UniversalDecimalInator."""

from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)


def strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def fetch_url(url: str, user_agent: str = "UniversalDecimalInator/1.0") -> str:
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    if "text/html" in resp.headers.get("Content-Type", ""):
        return clean_text(strip_html(resp.text))
    return clean_text(resp.text)
