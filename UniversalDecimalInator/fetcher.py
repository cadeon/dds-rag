"""URL fetching and content extraction for UniversalDecimalInator.

Supports:
  - HTML: BeautifulSoup + readability for article extraction
  - PDF: pymupdf (PyMuPDF) for text extraction
  - Plain text: passthrough with cleanup
"""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _try_readability(html: str) -> str | None:
    """Try to extract article content using readability.

    Returns None if readability is not available or fails.
    """
    try:
        from readability import Document
        doc = Document(html)
        return doc.summary()
    except Exception as e:
        logger.debug("readability extraction failed: %s", e)
        return None


def extract_html(html: str, url: str = "") -> str:
    """Extract readable text from HTML content.

    Tries readability first for article-focused extraction, falls back to
    BeautifulSoup with aggressive noise removal.
    """
    # Try readability for article content
    summary = _try_readability(html)
    if summary:
        soup = BeautifulSoup(summary, "html.parser")
        # Remove nav/header/footer/aside from readability output too
        for tag_name in ["nav", "header", "footer", "aside"]:
            for element in soup.find_all(tag_name):
                element.decompose()
        # Preserve img alt text
        for img in soup.find_all("img"):
            alt = img.get("alt", "")
            if isinstance(alt, list):
                alt = alt[0] if alt else ""
            alt = str(alt).strip()
            if alt:
                img.replace_with(alt)
            else:
                img.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return clean_text(text)

    # Fallback: BeautifulSoup with noise removal
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, and other noise elements
    for element in soup.find_all(["script", "style", "noscript", "iframe", "svg", "noscript"]):
        element.decompose()

    # Remove nav, header, footer, sidebar, ads
    for tag_name in ["nav", "header", "footer", "aside"]:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove common ad/tracking classes
    for element in soup.find_all(class_=re.compile(r"ad|menu|sidebar|social|share|cookie|popup|modal", re.I)):
        element.decompose()

    # Handle images — preserve alt text before removing
    for img in soup.find_all("img"):
        alt = img.get("alt", "")
        if isinstance(alt, list):
            alt = alt[0] if alt else ""
        alt = str(alt).strip()
        if alt:
            img.replace_with(alt)
        else:
            img.decompose()

    # Remove links (keep text)
    for a in soup.find_all("a"):
        a.replace_with(a.text)

    # Extract text with sensible separators
    text = soup.get_text(separator="\n", strip=True)
    return clean_text(text)


def extract_pdf(pdf_data: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install with: pip install pymupdf"
        )

    doc = fitz.open(stream=pdf_data, filetype="pdf")
    pages = []
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            pages.append(page_text.strip())
    doc.close()

    return clean_text("\n\n".join(pages))


def clean_text(text: str) -> str:
    """Normalize whitespace and clean extracted text."""
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Strip each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def fetch_url(url: str, user_agent: str = "UniversalDecimalInator/1.0") -> str:
    """Fetch a URL and extract readable text content.

    Auto-detects content type and uses the appropriate extractor:
    - text/html: extract_html()
    - application/pdf: extract_pdf()
    - text/plain or other: clean_text()
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/pdf,application/xhtml+xml,*/*",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()

    if "application/pdf" in content_type:
        return extract_pdf(resp.content)
    elif "text/html" in content_type:
        return extract_html(resp.text, url)
    else:
        return clean_text(resp.text)


# Legacy aliases for backward compatibility
strip_html = extract_html
