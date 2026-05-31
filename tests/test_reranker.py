"""Tests for dds_rag.reranker — Reranker class."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.reranker import Reranker


@pytest.fixture(scope="module")
def reranker():
    return Reranker(model="BAAI/bge-reranker-base")


class TestRerankerRerank:
    def test_rerank_returns_indexed_scores(self, reranker):
        query = "What is Python programming?"
        docs = [
            "Python is a high-level programming language known for readability.",
            "The weather today is sunny and warm.",
            "Python programming involves writing code in the Python language.",
        ]
        result = reranker.rerank(query, docs)
        assert len(result) == 3
        # Each item is (original_index, score)
        for idx, score in result:
            assert 0 <= idx < 3
            assert isinstance(score, (int, float))

    def test_rerank_relevant_ranks_higher(self, reranker):
        query = "What is Python programming?"
        docs = [
            "Python is a high-level programming language known for readability.",
            "The weather today is sunny and warm.",
        ]
        result = reranker.rerank(query, docs)
        # Most relevant doc should be first
        assert result[0][0] == 0  # Python doc should rank above weather

    def test_rerank_top_k(self, reranker):
        query = "What is Python?"
        docs = [f"Document {i}" for i in range(10)]
        result = reranker.rerank(query, docs, top_k=3)
        assert len(result) == 3

    def test_rerank_empty_docs(self, reranker):
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_single_doc(self, reranker):
        result = reranker.rerank("query", ["single document"])
        assert len(result) == 1
        assert result[0][0] == 0


class TestRerankerLongDocuments:
    def test_long_document_no_crash(self, reranker):
        # Very long document — should not crash (may be truncated internally)
        long_doc = "Word. " * 3000
        result = reranker.rerank("query", [long_doc])
        assert len(result) == 1

    def test_very_long_document(self, reranker):
        # Extreme length — beyond typical model context
        extreme_doc = "Word. " * 10000
        result = reranker.rerank("query", [extreme_doc])
        assert len(result) == 1


class TestRerankerEdgeCases:
    def test_special_characters(self, reranker):
        result = reranker.rerank(
            "Query with <special> & chars!",
            ["Document with 'quotes' and \"double quotes\""],
        )
        assert len(result) == 1

    def test_unicode(self, reranker):
        result = reranker.rerank("查询测试", ["这是一个测试文档"])
        assert len(result) == 1

    def test_mixed_relevance(self, reranker):
        query = "machine learning algorithms"
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "The cat sat on the mat.",
            "Algorithms are step-by-step procedures for calculations.",
            "Deep learning uses neural networks with many layers.",
        ]
        result = reranker.rerank(query, docs)
        # Relevant docs (0, 2, 3) should rank above irrelevant (1)
        relevant = {idx for idx, _ in result if idx in {0, 2, 3}}
        irrelevant = {idx for idx, _ in result if idx == 1}
        # At least one relevant doc should appear before the irrelevant one
        relevant_positions = [i for i, (idx, _) in enumerate(result) if idx in {0, 2, 3}]
        irrelevant_positions = [i for i, (idx, _) in enumerate(result) if idx == 1]
        if relevant_positions and irrelevant_positions:
            assert min(relevant_positions) < max(irrelevant_positions)
