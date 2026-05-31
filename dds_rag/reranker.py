"""Cross-encoder reranker for final precision ranking."""

from __future__ import annotations

import requests

CROSS_ENCODER_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    pass


class Reranker:
    """Rerank query-chunk pairs using a cross-encoder model."""

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
                self._mode = "none"
        elif endpoint:
            self._mode = "remote"
        else:
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

    def _rerank_local(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
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
            "documents": texts,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        scores = []
        for item in data.get("results", data.get("data", [])):
            scores.append(item.get("relevance_score", item.get("score", 0.0)))
        return scores
