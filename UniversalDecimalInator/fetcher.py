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


def _is_noise_image(src: str, alt: str, img_tag) -> bool:
    """Filter out tracking pixels, icons, and other non-content images."""
    # Skip data URIs (usually icons/tracking)
    if src.startswith("data:"):
        return True
    # Skip math equation renders from Wikimedia
    if "wikimedia.org/api/rest_v1/media/math/render" in src:
        return True
    # Skip known noise patterns in URLs
    noise_patterns = [
        "CentralAutoLogin", "1x1", "pixel", "tracking", "beacon",
        "favicon", "gravatar",
        "analytics", "google-analytics",
    ]
    src_lower = src.lower()
    if any(p in src_lower for p in noise_patterns):
        return True
    # Skip images with very short alt text that look like UI elements
    if alt and len(alt) <= 3 and alt.isalpha():
        return True
    # Skip images with width/height attributes indicating tiny images
    if img_tag:
        for attr in ["width", "height"]:
            val = img_tag.get(attr, "")
            if isinstance(val, str):
                try:
                    if int(val) <= 5:
                        return True
                except ValueError:
                    pass
    return False


def extract_html(html: str, url: str = "") -> tuple[str, list[dict]]:
    """Extract readable text and image info from HTML content.

    Tries readability first for article-focused extraction, falls back to
    BeautifulSoup with aggressive noise removal.

    Returns (text, images) where images is a list of dicts with keys:
      - src: image URL (absolute)
      - alt: alt text
      - content_type: inferred MIME type
    """
    images = []

    # Try readability for article content
    summary = _try_readability(html)
    if summary:
        soup = BeautifulSoup(summary, "html.parser")
        # Remove nav/header/footer/aside from readability output too
        for tag_name in ["nav", "header", "footer", "aside"]:
            for element in soup.find_all(tag_name):
                element.decompose()
        # Extract images before removing them
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if isinstance(src, list):
                src = src[0] if src else ""
            src = str(src).strip()
            if src:
                # Make absolute URL
                src = _make_absolute(src, url)
                alt = img.get("alt", "")
                if isinstance(alt, list):
                    alt = alt[0] if alt else ""
                alt = str(alt).strip()
                # Skip noise images
                if not _is_noise_image(src, alt, img):
                    images.append({
                        "src": src,
                        "alt": alt,
                        "content_type": _guess_image_type(src),
                    })
            alt = img.get("alt", "")
            if isinstance(alt, list):
                alt = alt[0] if alt else ""
            alt = str(alt).strip()
            if alt:
                img.replace_with(alt)
            else:
                img.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return clean_text(text), images

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

    # Extract images before removing
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if isinstance(src, list):
            src = src[0] if src else ""
        src = str(src).strip()
        if src:
            src = _make_absolute(src, url)
            alt = img.get("alt", "")
            if isinstance(alt, list):
                alt = alt[0] if alt else ""
            alt = str(alt).strip()
            if not _is_noise_image(src, alt, img):
                images.append({
                    "src": src,
                    "alt": alt,
                    "content_type": _guess_image_type(src),
                })
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
    return clean_text(text), images


def _make_absolute(url: str, base_url: str) -> str:
    """Convert relative URL to absolute."""
    if url.startswith(("http://", "https://", "data:")):
        return url
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        # Extract scheme and host from base
        if base_url.startswith("https://"):
            return "https://" + base_url.split("//", 1)[1].split("/", 1)[0] + url
        elif base_url.startswith("http://"):
            return "http://" + base_url.split("//", 1)[1].split("/", 1)[0] + url
    else:
        # Relative path
        base_dir = base_url.rsplit("/", 1)[0]
        return base_dir + "/" + url
    return url


def _guess_image_type(url: str) -> str:
    """Guess MIME type from URL extension."""
    ext = url.rsplit(".", 1)[-1].lower().split("?")[0]
    types = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "svg": "image/svg+xml",
        "bmp": "image/bmp", "tiff": "image/tiff",
    }
    return types.get(ext, "image/jpeg")


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


def fetch_url(url: str, user_agent: str = "UniversalDecimalInator/1.0") -> tuple[str, list[dict]]:
    """Fetch a URL and extract readable text content.

    Auto-detects content type and uses the appropriate extractor:
    - text/html: extract_html()
    - application/pdf: extract_pdf()
    - text/plain or other: clean_text()

    Returns (text, images) where images is a list of image info dicts
    (empty list for non-HTML content).
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/pdf,application/xhtml+xml,*/*",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()

    if "application/pdf" in content_type:
        return extract_pdf(resp.content), []
    elif "text/html" in content_type:
        return extract_html(resp.text, url)
    else:
        return clean_text(resp.text), []


# Legacy alias for backward compatibility — returns (text, images) tuple now.
strip_html = extract_html
