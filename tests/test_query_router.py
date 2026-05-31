"""Tests for dds_rag.query_router — QueryRouter and RRF."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.query_router import QueryRouter, reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_fusion_single_list(self):
        lists = [
            [("a", 0.9), ("b", 0.8), ("c", 0.7)],
        ]
        result = reciprocal_rank_fusion(lists)
        assert result[0][0] == "a"
        assert result[0][1] > result[1][1]

    def test_fusion_agrees_rank_higher(self):
        lists = [
            [("a", 0.9), ("b", 0.8)],
            [("a", 0.95), ("c", 0.7)],
        ]
        result = reciprocal_rank_fusion(lists)
        # "a" is rank 1 in both lists → should be highest
        assert result[0][0] == "a"

    def test_fusion_disagree(self):
        lists = [
            [("a", 0.9), ("b", 0.8)],
            [("b", 0.9), ("a", 0.8)],
        ]
        result = reciprocal_rank_fusion(lists)
        # Both at rank 1 → tied, but "a" appears first in first list
        assert result[0][0] in ("a", "b")
        assert len(result) == 2

    def test_fusion_empty_lists(self):
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_fusion_with_empty_list(self):
        lists = [
            [("a", 0.9)],
            [],
        ]
        result = reciprocal_rank_fusion(lists)
        assert len(result) == 1
        assert result[0][0] == "a"

    def test_fusion_custom_k(self):
        lists = [
            [("a", 1.0), ("b", 0.5), ("c", 0.3)],
        ]
        result_default = reciprocal_rank_fusion(lists, k=60)
        result_small = reciprocal_rank_fusion(lists, k=1)
        # Same ranking order regardless of k
        assert [r[0] for r in result_default] == [r[0] for r in result_small]


class TestQueryRouterInit:
    def test_defaults(self):
        router = QueryRouter()
        assert router.min_results_threshold == 10
        assert router.fallback_parent is True
        assert router.expand_siblings is True
        assert router.min_confidence == 0.7

    def test_custom_params(self):
        router = QueryRouter(
            min_results_threshold=5,
            fallback_parent=False,
            expand_siblings=False,
            min_confidence=0.5,
        )
        assert router.min_results_threshold == 5
        assert router.fallback_parent is False
        assert router.expand_siblings is False
        assert router.min_confidence == 0.5
