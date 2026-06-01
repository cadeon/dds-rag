"""Tests for UniversalDecimalInator.fetcher — URL fetching and content extraction."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.fetcher import extract_html, clean_text


class TestExtractHTML:
    def test_basic_tags(self):
        result = extract_html("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result

    def test_script_removal(self):
        result = extract_html("<p>Text</p><script>alert('xss')</script>")
        assert "alert" not in result
        assert "Text" in result

    def test_style_removal(self):
        result = extract_html("<p>Content</p><style>.red { color: red; }</style>")
        assert "color" not in result
        assert "Content" in result

    def test_nav_removal(self):
        html = "<nav><a href='/'>Home</a><a href='/about'>About</a></nav><main><p>Article body</p></main>"
        result = extract_html(html)
        assert "Article body" in result
        assert "Home" not in result

    def test_img_alt_preserved(self):
        result = extract_html("<p>See <img alt='diagram of process'> below</p>")
        assert "diagram of process" in result

    def test_link_text_preserved(self):
        result = extract_html("<p>Read <a href='https://example.com'>the docs</a> for more</p>")
        assert "the docs" in result
        assert "https://example.com" not in result

    def test_iframe_removal(self):
        result = extract_html("<p>Text</p><iframe src='https://ads.example.com'></iframe>")
        assert "ads.example.com" not in result
        assert "Text" in result


class TestCleanText:
    def test_remove_extra_newlines(self):
        result = clean_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strip_whitespace(self):
        result = clean_text("  hello  ")
        assert result == "hello"

    def test_collapse_spaces(self):
        result = clean_text("hello    world")
        assert result == "hello world"

    def test_strip_lines(self):
        result = clean_text("  line1  \n  line2  ")
        assert "  line1" not in result
        assert "line1" in result
