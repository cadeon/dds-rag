"""Query router - hybrid search with DDC routing, BM25, vector, and RRF."""

from __future__ import annotations

import math

from dds_rag.ddc import get_parent, get_siblings
from dds_rag.models import Chunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    Each list is (id, score) pairs. Returns (id, fused_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list):
            score = 1.0 / (k + rank + 1)
            scores[item_id] = scores.get(item_id, 0.0) + score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class QueryRouter:
    """Routes queries through the card catalog using hybrid search."""

    def __init__(
        self,
        min_results_threshold: int = 10,
        fallback_parent: bool = True,
        expand_siblings: bool = True,
        min_confidence: float = 0.7,
    ):
        self.min_results_threshold = min_results_threshold
        self.fallback_parent = fallback_parent
        self.expand_siblings = expand_siblings
        self.min_confidence = min_confidence

    def route(
        self,
        query: str,
        query_vec: list[float],
        storage,
        reranker,
        top_k: int = 10,
        fused_top_k: int = 20,
    ) -> list[Chunk]:
        """
        Execute the full query pipeline:
        1. Find candidate DDC branch via keyword matching
        2. Hybrid search (BM25 + vector) on candidates
        3. RRF fusion
        4. Pull chunks
        5. Rerank
        """
        # Phase 1: Find candidate cards via keyword matching on tags/topics
        candidate_cards = self._find_candidates(query, storage)

        if not candidate_cards:
            # No candidates found, search full corpus
            candidate_cards = storage.list_cards(limit=1000)

        candidate_ids = [c.id for c in candidate_cards]

        # Phase 2: Hybrid search
        bm25_results = storage.bm25_search(query, candidate_ids)
        vector_results = storage.vector_search(query_vec, candidate_ids)

        # Phase 3: RRF fusion
        fused = reciprocal_rank_fusion([bm25_results, vector_results])

        # Phase 4: Fallback if too few results
        if len(fused) < self.min_results_threshold:
            fused = self._fallback_search(query, query_vec, candidate_cards, storage, fused)

        # Phase 5: Pull chunks for top cards
        top_card_ids = [cid for cid, _ in fused[:fused_top_k]]
        chunks = storage.pull_chunks(top_card_ids)

        # Phase 6: Rerank
        if chunks and reranker:
            chunk_texts = [c.text for c in chunks]
            reranked = reranker.rerank(query, chunk_texts, top_k=top_k)
            # Map back to chunks
            result_chunks = [chunks[idx] for idx, _ in reranked]
        else:
            result_chunks = chunks[:top_k]

        return result_chunks

    def route_ddc(
        self,
        query: str,
        query_vec: list[float],
        ddc_number: float,
        storage,
        reranker,
        top_k: int = 10,
        fused_top_k: int = 20,
    ) -> list[Chunk]:
        """Search within a specific DDC branch."""
        candidate_cards = storage.cards_by_ddc(ddc_number)
        candidate_ids = [c.id for c in candidate_cards]

        bm25_results = storage.bm25_search(query, candidate_ids)
        vector_results = storage.vector_search(query_vec, candidate_ids)
        fused = reciprocal_rank_fusion([bm25_results, vector_results])

        top_card_ids = [cid for cid, _ in fused[:fused_top_k]]
        chunks = storage.pull_chunks(top_card_ids)

        if chunks and reranker:
            chunk_texts = [c.text for c in chunks]
            reranked = reranker.rerank(query, chunk_texts, top_k=top_k)
            result_chunks = [chunks[idx] for idx, _ in reranked]
        else:
            result_chunks = chunks[:top_k]

        return result_chunks

    def _find_candidates(self, query: str, storage) -> list:
        """Find candidate cards by matching query terms against tags/topics."""
        query_terms = set(query.lower().split())
        query_terms = {t for t in query_terms if len(t) > 2}  # Filter short words

        if not query_terms:
            return []

        all_cards = storage.list_cards(limit=5000)
        scored: dict[str, float] = {}

        for card in all_cards:
            score = 0.0
            card_tags = set(t.lower() for t in card.tags)
            card_topics = set(t.lower() for t in card.topics)
            card_abstract = card.abstract.lower()

            for term in query_terms:
                # Exact tag match (highest weight)
                if term in card_tags:
                    score += 10.0
                # Topic match
                elif any(term in topic for topic in card_topics):
                    score += 5.0
                # Abstract match
                elif term in card_abstract:
                    score += 1.0

            if score > 0:
                scored[card.id] = score

        # Return cards sorted by relevance score
        sorted_ids = sorted(scored.keys(), key=lambda x: scored[x], reverse=True)
        return [c for c in all_cards if c.id in sorted_ids]

    def _fallback_search(
        self,
        query: str,
        query_vec: list[float],
        initial_candidates: list,
        storage,
        current_fused: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fallback: widen search if too few results."""
        if not initial_candidates:
            # Full corpus search
            all_cards = storage.list_cards(limit=10000)
            bm25 = storage.bm25_search(query)
            vec = storage.vector_search(query_vec)
            return reciprocal_rank_fusion([bm25, vec])

        # Try parent class
        if self.fallback_parent:
            parents = set()
            for card in initial_candidates:
                parents.add(card.ddc_parent)
            for parent in parents:
                parent_cards = storage.cards_by_ddc_parent(parent)
                if len(parent_cards) > len(initial_candidates):
                    candidate_ids = [c.id for c in parent_cards]
                    bm25 = storage.bm25_search(query, candidate_ids)
                    vec = storage.vector_search(query_vec, candidate_ids)
                    fused = reciprocal_rank_fusion([bm25, vec])
                    if len(fused) >= self.min_results_threshold:
                        return fused

        # Try siblings
        if self.expand_siblings:
            for card in initial_candidates:
                for cls in card.ddc_classifications:
                    if cls.confidence >= self.min_confidence:
                        siblings = get_siblings(cls.number)
                        for sib in siblings:
                            sib_cards = storage.cards_by_ddc(sib)
                            if sib_cards:
                                candidate_ids = [c.id for c in sib_cards]
                                bm25 = storage.bm25_search(query, candidate_ids)
                                vec = storage.vector_search(query_vec, candidate_ids)
                                fused = reciprocal_rank_fusion([bm25, vec])
                                if fused:
                                    # Merge with current results
                                    merged = reciprocal_rank_fusion([current_fused, fused])
                                    if len(merged) >= self.min_results_threshold:
                                        return merged

        # Full corpus as last resort
        bm25 = storage.bm25_search(query)
        vec = storage.vector_search(query_vec)
        return reciprocal_rank_fusion([bm25, vec])
