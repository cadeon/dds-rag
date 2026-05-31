"""Cross-encoder reranker for final precision ranking."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

CROSS_ENCODER_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    pass


class Reranker:
    """Rerank query-chunk pairs using a cross-encoder model."""

    # BAAI/bge-reranker-base has a max sequence length of 512
    # Reserve 64 tokens for the query, leave 448 for documents
    MAX_DOCUMENT_LENGTH = 448

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        endpoint: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model
        self.endpoint = endpoint
        self.api_key = api_key
        self._local_model = None

        if CROSS_ENCODER_AVAILABLE and not endpoint:
            try:
                self._local_model = CrossEncoder(model)
                self._mode = "local"
            except Exception:
                logger.warning("Failed to load cross-encoder model '%s' — reranking disabled", model)
                self._mode = "none"
        elif endpoint:
            self._mode = "remote"
        else:
            logger.warning("No reranker backend available — reranking disabled")
            self._mode = "none"

    def rerank(
        self,
        query: str,
        texts: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank texts against a query.
        Returns list of (original_index, score) sorted by score descending.
        """
        if not texts:
            return []

        if self._mode == "local":
            scores = self._rerank_local(query, texts)
        elif self._mode == "remote":
            scores = self._rerank_remote(query, texts)
        else:
            # Fallback: return in original order with equal scores
            scores = [0.0] * len(texts)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed = indexed[:top_k]

        return indexed

    @classmethod
    def _truncate(cls, text: str, max_chars: int = 2000) -> str:
        """Truncate text to fit within model context window.

        ~448 tokens ≈ 2000 chars for typical English text.
        Preserves sentence boundaries when possible.
        """
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # Try to cut at sentence boundary
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.5:
            truncated = truncated[:last_period + 1]
        return truncated

    def _rerank_local(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, self._truncate(text)] for text in texts]
        scores = self._local_model.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    def _rerank_remote(self, query: str, texts: list[str]) -> list[float]:
        """Call remote reranking endpoint (Cohere-style)."""
        url = self.endpoint
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [self._truncate(t) for t in texts],
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        scores = []
        for item in data.get("results", data.get("data", [])):
            scores.append(item.get("relevance_score", item.get("score", 0.0)))
        return scores
