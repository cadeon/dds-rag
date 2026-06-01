"""Tests for UniversalDecimalInator.fetcher — URL fetching and content extraction."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.fetcher import extract_html, clean_text, _make_absolute, _guess_image_type


class TestExtractHTML:
    def test_basic_tags(self):
        text, images = extract_html("<p>Hello <b>world</b></p>")
        assert "Hello" in text
        assert "world" in text
        assert isinstance(images, list)

    def test_script_removal(self):
        text, _ = extract_html("<p>Text</p><script>alert('xss')</script>")
        assert "alert" not in text
        assert "Text" in text

    def test_style_removal(self):
        text, _ = extract_html("<p>Content</p><style>.red { color: red; }</style>")
        assert "color" not in text
        assert "Content" in text

    def test_nav_removal(self):
        html = "<nav><a href='/'>Home</a><a href='/about'>About</a></nav><main><p>Article body</p></main>"
        result, _ = extract_html(html)
        assert "Article body" in result
        assert "Home" not in result

    def test_img_alt_preserved(self):
        text, _ = extract_html("<p>See <img alt='diagram of process'> below</p>")
        assert "diagram of process" in text

    def test_img_extracted(self):
        html = "<p><img src='https://example.com/photo.jpg' alt='A river'></p>"
        text, images = extract_html(html, url="https://example.com/page")
        assert len(images) == 1
        assert images[0]["src"] == "https://example.com/photo.jpg"
        assert images[0]["alt"] == "A river"
        assert images[0]["content_type"] == "image/jpeg"

    def test_img_relative_url(self):
        html = "<p><img src='/images/logo.png'></p>"
        text, images = extract_html(html, url="https://example.com/page")
        assert len(images) == 1
        assert images[0]["src"] == "https://example.com/images/logo.png"

    def test_link_text_preserved(self):
        text, _ = extract_html("<p>Read <a href='https://example.com'>the docs</a> for more</p>")
        assert "the docs" in text
        assert "https://example.com" not in text

    def test_iframe_removal(self):
        text, _ = extract_html("<p>Text</p><iframe src='https://ads.example.com'></iframe>")
        assert "ads.example.com" not in text
        assert "Text" in text


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


class TestMakeAbsolute:
    def test_absolute_url_unchanged(self):
        assert _make_absolute("https://example.com/img.jpg", "https://example.com/page") == "https://example.com/img.jpg"

    def test_protocol_relative(self):
        assert _make_absolute("//cdn.example.com/img.jpg", "https://example.com/page") == "https://cdn.example.com/img.jpg"

    def test_root_relative(self):
        assert _make_absolute("/images/img.jpg", "https://example.com/page") == "https://example.com/images/img.jpg"

    def test_relative_path(self):
        assert _make_absolute("img.jpg", "https://example.com/page/sub/") == "https://example.com/page/sub/img.jpg"

    def test_data_url_unchanged(self):
        assert _make_absolute("data:image/png;base64,abc", "https://example.com/page") == "data:image/png;base64,abc"


class TestGuessImageType:
    def test_jpg(self):
        assert _guess_image_type("photo.jpg") == "image/jpeg"

    def test_png(self):
        assert _guess_image_type("logo.png") == "image/png"

    def test_webp(self):
        assert _guess_image_type("banner.webp") == "image/webp"

    def test_query_string(self):
        assert _guess_image_type("photo.jpg?v=2") == "image/jpeg"

    def test_unknown(self):
        assert _guess_image_type("photo.xyz") == "image/jpeg"
