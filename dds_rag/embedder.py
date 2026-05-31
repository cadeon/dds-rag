"""Embedding service - supports both local sentence-transformers and remote API."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import requests

# Try to import sentence-transformers for local embeddings
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class Embedder:
    """Embed text into vectors. Supports local models and remote APIs."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        endpoint: str | None = None,
        dimensions: int = 384,
        api_key: str | None = None,
    ):
        self.model_name = model
        self.endpoint = endpoint
        self.dimensions = dimensions
        self.api_key = api_key
        self._local_model = None

        # Try local model first
        if SENTENCE_TRANSFORMERS_AVAILABLE and not endpoint:
            try:
                self._local_model = SentenceTransformer(model)
                self.dimensions = self._local_model.get_sentence_embedding_dimension()
                self._mode = "local"
            except Exception:
                self._mode = "none"
        elif endpoint:
            self._mode = "remote"
        else:
            self._mode = "none"

        if self._mode == "none":
            raise RuntimeError(
                "No embedding backend available. "
                "Install sentence-transformers or configure an embedding endpoint."
            )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        texts = [text]
        return self.embed_batch(texts)[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if self._mode == "local":
            return self._embed_local(texts)
        elif self._mode == "remote":
            return self._embed_remote(texts)
        else:
            raise RuntimeError("No embedding backend available")

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._local_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        """Call remote OpenAI-compatible embedding endpoint."""
        url = self.endpoint.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "input": texts,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        embeddings = []
        for item in data["data"]:
            embeddings.append(item["embedding"])
        return embeddings

    @property
    def dim(self) -> int:
        return self.dimensions
