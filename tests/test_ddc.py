"""Tests for dds_rag.ddc — DDC classification utilities."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.ddc import DDC_CLASSES, MAIN_CLASSES, get_ddc_tree, get_label, get_main_class


class TestMainClasses:
    def test_main_classes_count(self):
        # 10 main classes in DDC
        assert len(MAIN_CLASSES) == 10

    def test_main_class_0(self):
        assert 0 in MAIN_CLASSES
        assert "Computer" in MAIN_CLASSES[0]

    def test_main_class_5(self):
        assert 500 in MAIN_CLASSES
        assert "science" in MAIN_CLASSES[500].lower()

    def test_main_class_9(self):
        assert 900 in MAIN_CLASSES
        assert "History" in MAIN_CLASSES[900]


class TestDDCClasses:
    def test_has_entries(self):
        assert len(DDC_CLASSES) > 0

    def test_known_class(self):
        # 530 is Physics
        assert 530 in DDC_CLASSES

    def test_class_label(self):
        label = DDC_CLASSES[530]
        assert isinstance(label, str)
        assert len(label) > 0


class TestGetLabel:
    def test_known_label(self):
        label = get_label(530)
        assert label is not None
        assert len(label) > 0

    def test_unknown_label(self):
        label = get_label(999.99)
        # Should return something reasonable, not crash
        assert label is not None


class TestGetMainClass:
    def test_530_returns_500(self):
        assert get_main_class(530.12) == 500

    def test_005_returns_0(self):
        assert get_main_class(5.0) == 0

    def test_909_returns_900(self):
        assert get_main_class(909.0) == 900


class TestGetDDCTree:
    def test_returns_tree(self):
        tree = get_ddc_tree()
        assert isinstance(tree, dict)
        assert len(tree) > 0

    def test_tree_has_main_classes(self):
        tree = get_ddc_tree()
        for main_class in MAIN_CLASSES:
            assert main_class in tree
