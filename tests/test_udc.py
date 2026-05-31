"""Tests for UniversalDecimalInator.reference — ClassificationReference loader."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.reference import ClassificationReference

UDC_REF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "udc_reference.yaml",
)


@pytest.fixture
def ref():
    return ClassificationReference(UDC_REF)


class TestClassificationReference:
    def test_loads(self, ref):
        assert ref.name == "UDC"
        assert ref.version == "1.0"

    def test_has_main_classes(self, ref):
        assert len(ref.main_classes) == 10

    def test_main_class_0(self, ref):
        assert "0" in ref.main_classes
        label = ref.main_classes["0"].lower()
        assert "computer" in label or "information" in label

    def test_has_classes(self, ref):
        assert len(ref.classes) > 100

    def test_get_label(self, ref):
        label = ref.get_label("519.684")
        assert label is not None

    def test_get_main_class(self, ref):
        assert ref.get_main_class("519.684") == "5"

    def test_004_returns_0(self, ref):
        assert ref.get_main_class("004.738.5") == "0"

    def test_compound_returns_first(self, ref):
        assert ref.get_main_class("004.738.5:179.4") == "0"

    def test_standard_subdivisions(self, ref):
        assert len(ref.standard_subdivisions) > 0

    def test_geographic_table(self, ref):
        assert len(ref.geographic_table) > 0

    def test_syntax_rules(self, ref):
        rules = ref.syntax_rules
        assert isinstance(rules, list)
        assert len(rules) > 0


class TestParseCompound:
    def test_simple_compound(self, ref):
        result = ref.parse_compound("004.738.5:179.4:616")
        assert result == ["004.738.5", "179.4", "616"]

    def test_single_component(self, ref):
        result = ref.parse_compound("519.684")
        assert result == ["519.684"]

    def test_empty(self, ref):
        result = ref.parse_compound("")
        assert result == [""]


class TestFirstComponent:
    def test_compound(self, ref):
        assert ref.first_component("004.738.5:179.4") == "004.738.5"

    def test_simple(self, ref):
        assert ref.first_component("519.684") == "519.684"


class TestStripFacets:
    def test_with_standard_subdivision(self, ref):
        assert ref.strip_facets("621.396-055.2") == "621.396"

    def test_with_geographic(self, ref):
        assert ref.strip_facets("430(430)") == "430"

    def test_both(self, ref):
        assert ref.strip_facets("621.396-055.2(430)") == "621.396"

    def test_clean(self, ref):
        assert ref.strip_facets("519.684") == "519.684"


class TestFormatCompound:
    def test_with_labels(self, ref):
        result = ref.format_compound(
            ["004.738.5", "179.4"],
            ["Machine learning", "Bioethics"]
        )
        assert "004.738.5" in result
        assert "179.4" in result
        assert "Machine learning" in result
        assert "Bioethics" in result

    def test_without_labels(self, ref):
        result = ref.format_compound(["004.738.5", "179.4"], None)
        assert "004.738.5:179.4" == result


class TestValidate:
    def test_valid(self, ref):
        assert ref.validate("519.684") is True
        assert ref.validate("004.738.5:179.4") is True
        assert ref.validate("621.396-055.2(430)") is True

    def test_invalid(self, ref):
        assert ref.validate("") is False
        assert ref.validate("abc") is False
        assert ref.validate("519.684!") is False


class TestGetUDCTree:
    def test_returns_tree(self, ref):
        tree = ref.get_tree()
        assert isinstance(tree, dict)
        assert len(tree) > 0

    def test_tree_has_main_classes(self, ref):
        tree = ref.get_tree()
        for mc in ref.main_classes:
            assert mc in tree


class TestGetChildren:
    def test_root_children(self, ref):
        children = ref.get_children("5")
        assert isinstance(children, list)

    def test_no_children(self, ref):
        children = ref.get_children("999")
        assert children == []


class TestGetAncestors:
    def test_ancestors(self, ref):
        ancestors = ref.get_ancestors("519.684")
        assert len(ancestors) > 0


class TestUDCPath:
    def test_simple_udc(self, ref):
        path = ref.udc_to_path("516.3")
        assert path == "5/516/516.3"

    def test_compound_udc(self, ref):
        path = ref.udc_to_path("004.738.5:179.4")
        assert path == "0/004/004.738.5"

    def test_with_facets(self, ref):
        path = ref.udc_to_path("621.396-055.2(430)")
        assert path == "6/621/621.396-055.2(430)"


class TestSystemPrompt:
    def test_system_prompt(self, ref):
        prompt = ref.system_prompt()
        assert "Universal Decimal Classification" in prompt
        assert "519.684" in prompt
