"""Data models for UniversalDecimalInator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


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
class Card:
    """A knowledge base card with UDC classification."""

    id: str
    title: str = ""
    abstract: str = ""
    classification: UDCClassification = field(default_factory=lambda: UDCClassification(primary=""))
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    source_url: str = ""
    author: str = ""
    created_at: str = ""
    updated_at: str = ""
    content: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "classification": self.classification.to_dict(),
            "tags": self.tags,
            "topics": self.topics,
            "source_url": self.source_url,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            classification=UDCClassification.from_dict(d.get("classification", {})),
            tags=d.get("tags", []),
            topics=d.get("topics", []),
            source_url=d.get("source_url", ""),
            author=d.get("author", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            content=d.get("content", ""),
        )
