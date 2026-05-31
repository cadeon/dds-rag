"""Tests for dds_rag.embedder — Embedder class."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder(model="sentence-transformers/all-MiniLM-L6-v2")


class TestEmbedderEmbed:
    def test_single_embedding(self, embedder):
        vec = embedder.embed("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384  # MiniLM default dimension
        # Should be normalized (unit vector)
        norm = np.linalg.norm(vec)
        assert 0.99 < norm < 1.01

    def test_batch_embedding(self, embedder):
        texts = ["Hello", "World", "Python"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == 384

    def test_similar_text_similar_embedding(self, embedder):
        vec1 = embedder.embed("Python is a programming language")
        vec2 = embedder.embed("Python is a coding language")
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        assert similarity > 0.7  # semantically similar

    def test_dissimilar_text_dissimilar_embedding(self, embedder):
        vec1 = embedder.embed("Python programming language")
        vec2 = embedder.embed("Cooking a delicious pasta dinner")
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        assert similarity < 0.5  # semantically different

    def test_empty_string(self, embedder):
        vec = embedder.embed("")
        assert len(vec) == 384

    def test_large_batch(self, embedder):
        texts = [f"Document number {i} about various topics" for i in range(100)]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 100
        assert all(len(v) == 384 for v in vecs)
