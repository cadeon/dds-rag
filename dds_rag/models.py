"""Data models for DDS-RAG."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DDCClassification:
    """A Dewey Decimal Classification with confidence score."""
    number: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict) -> DDCClassification:
        return cls(number=float(d["number"]), confidence=float(d["confidence"]))


@dataclass
class Chunk:
    """A text chunk from a document."""
    id: str
    text: str
    embedding: list[float] | None = None
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            id=d["id"],
            text=d["text"],
            embedding=d.get("embedding"),
            offset=d.get("offset", 0),
        )


@dataclass
class Card:
    """A card catalog entry for a document."""
    id: str
    ddc_classifications: list[DDCClassification]
    ddc_parent: float
    abstract: str
    tags: list[str]
    topics: list[str]
    audience: str
    format: str
    date: str
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ddc_classifications": [c.to_dict() for c in self.ddc_classifications],
            "ddc_parent": self.ddc_parent,
            "abstract": self.abstract,
            "tags": self.tags,
            "topics": self.topics,
            "audience": self.audience,
            "format": self.format,
            "date": self.date,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        return cls(
            id=d["id"],
            ddc_classifications=[DDCClassification.from_dict(c) for c in d["ddc_classifications"]],
            ddc_parent=float(d["ddc_parent"]),
            abstract=d["abstract"],
            tags=d["tags"],
            topics=d["topics"],
            audience=d["audience"],
            format=d["format"],
            date=d["date"],
            embedding=d.get("embedding"),
        )


@dataclass
class Document:
    """A stored document linked to a card."""
    id: str
    card_id: str
    source: str
    chunks: list[Chunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "card_id": self.card_id,
            "source": self.source,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Document:
        return cls(
            id=d["id"],
            card_id=d["card_id"],
            source=d["source"],
            chunks=[Chunk.from_dict(c) for c in d.get("chunks", [])],
        )


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a vector to bytes for SQLite BLOB storage."""
    return json.dumps(vec).encode("utf-8")


def deserialize_vector(data: bytes) -> list[float]:
    """Deserialize a vector from bytes."""
    if data is None:
        return None
    return json.loads(data.decode("utf-8"))


def serialize_json(obj: Any) -> bytes:
    """Serialize any JSON-serializable object to bytes."""
    return json.dumps(obj).encode("utf-8")


def deserialize_json(data: bytes) -> Any:
    """Deserialize from bytes to Python object."""
    if data is None:
        return None
    return json.loads(data.decode("utf-8"))
