"""Tests for UniversalDecimalInator.fetcher — URL fetching and content extraction."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.fetcher import strip_html, clean_text


class TestStripHTML:
    def test_basic_tags(self):
        result = strip_html("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result

    def test_script_removal(self):
        result = strip_html("<p>Text</p><script>alert('xss')</script>")
        assert "alert" not in result
        assert "Text" in result


class TestCleanText:
    def test_remove_extra_newlines(self):
        result = clean_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strip_whitespace(self):
        result = clean_text("  hello  ")
        assert result == "hello"
