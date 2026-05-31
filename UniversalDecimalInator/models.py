"""Data models for UniversalDecimalInator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


CARD_FORMAT_VERSION = "1"


@dataclass
class UDCClassification:
    """UDC classification with compound support.

    A classification can be a simple number (e.g. "519.684") or a compound
    colon-separated expression (e.g. "004.738.5:179.4:616").
    """

    primary: str  # Full classification string, possibly compound
    secondary: list[str] = field(default_factory=list)  # Additional classifications
    udc_main_class: str = ""  # Main class identifier derived from primary for directory placement

    def __post_init__(self):
        if not self.udc_main_class and self.primary:
            self.udc_main_class = self.primary[0] if self.primary else ""

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "udc_main_class": self.udc_main_class,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UDCClassification:
        return cls(
            primary=d.get("primary", ""),
            secondary=d.get("secondary", []),
            udc_main_class=str(d.get("udc_main_class", "")),
        )


@dataclass
class Source:
    """A source document or artifact reference for a Card.

    A source can be a URL, a local file, or any other provenance reference.
    It may have associated artifact files (images, snapshots, extracted text).
    """

    type: str = "url"  # url, file, api, other
    uri: str = ""  # URL or file path
    content_type: str = ""  # MIME type (text/html, application/pdf, etc.)
    artifacts: list[str] = field(default_factory=list)  # Relative paths to artifact files

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "uri": self.uri,
        }
        if self.content_type:
            d["content_type"] = self.content_type
        if self.artifacts:
            d["artifacts"] = self.artifacts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Source:
        return cls(
            type=d.get("type", "url"),
            uri=d.get("uri", d.get("url", "")),
            content_type=d.get("content_type", ""),
            artifacts=d.get("artifacts", []),
        )


@dataclass
class Card:
    """A knowledge base card with UDC classification."""

    id: str
    title: str = ""
    abstract: str = ""
    classification: UDCClassification = field(default_factory=lambda: UDCClassification(primary=""))
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    author: str = ""
    created_at: str = ""
    updated_at: str = ""
    content: str = ""
    format: str = ""  # Ingest method: url_fetch, text_paste, artifact_drop, batch
    udc_label: str = ""  # Human-readable label for primary classification
    version: str = CARD_FORMAT_VERSION  # Card file format version

    # Deprecated: use sources instead. Maintained for backward compatibility.
    source_url: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        # Migrate legacy source_url to sources if needed
        if self.source_url and not self.sources:
            self.sources = [Source(type="url", uri=self.source_url)]

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "classification": self.classification.to_dict(),
            "tags": self.tags,
            "topics": self.topics,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content": self.content,
        }
        if self.sources:
            d["sources"] = [s.to_dict() for s in self.sources]
        if self.format:
            d["format"] = self.format
        if self.udc_label:
            d["udc_label"] = self.udc_label
        if self.version:
            d["version"] = self.version
        # Keep source_url for backward compat
        if self.source_url:
            d["source_url"] = self.source_url
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        sources = []
        raw_sources = d.get("sources", [])
        if raw_sources:
            for s in raw_sources:
                if isinstance(s, dict):
                    sources.append(Source.from_dict(s))
                elif isinstance(s, str):
                    sources.append(Source(type="url", uri=s))
        # Backward compat: migrate source_url to sources
        source_url = d.get("source_url", "")
        if source_url and not sources:
            sources.append(Source(type="url", uri=source_url))
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            classification=UDCClassification.from_dict(d.get("classification", {})),
            tags=d.get("tags", []),
            topics=d.get("topics", []),
            sources=sources,
            source_url=source_url,
            author=d.get("author", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            content=d.get("content", ""),
            format=d.get("format", ""),
            udc_label=d.get("udc_label", ""),
            version=d.get("version", CARD_FORMAT_VERSION),
        )
