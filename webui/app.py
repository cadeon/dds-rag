"""UniversalDecimalInator Card Catalog WebUI - Flask application."""

from __future__ import annotations

import json
import logging
import os
import sys

import requests
import yaml
from flask import Flask, render_template, request, redirect, url_for, g

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.reference import ClassificationReference
from UniversalDecimalInator.fs import (
    list_cards, search_cards, catalog_stats, read_card, write_card, card_to_markdown,
    cards_by_classification,
)
from UniversalDecimalInator.ingest import ingest, ingest_url
from UniversalDecimalInator.admin import delete_card, reclassify
from UniversalDecimalInator.models import Card, UDCClassification, Source

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

KB_PATH = os.environ.get("KB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb"))
os.makedirs(KB_PATH, exist_ok=True)

REF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "udc_reference.yaml")
ref = ClassificationReference(REF_PATH)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.context_processor
def utility_processor():
    """Make utility functions and common context available in all templates."""
    from flask import g
    return dict(card_to_markdown=card_to_markdown, ref=g.get('ref'), stats=g.get('stats'))

# Cache for catalog stats - invalidated on write operations
_stats_cache: dict | None = None


def get_stats() -> dict:
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = catalog_stats(KB_PATH)
    return _stats_cache


def invalidate_stats():
    global _stats_cache
    _stats_cache = None


@app.template_filter("udc_label")
def udc_label(num):
    return ref.get_label(str(num))


@app.before_request
def load_context():
    g.ref = ref
    g.stats = get_stats()


@app.route("/")
def index():
    main_classes = {k: v for k, v in ref.main_classes.items()}
    cards = list_cards(KB_PATH)[:12]
    return render_template("index.html", main_classes=main_classes, cards=cards, active_nav="catalog")


@app.route("/browse/<udc_number>")
def browse(udc_number):
    cards = cards_by_classification(KB_PATH, udc_number, ref)
    label = ref.get_label(udc_number)
    # Derive sub-classifications from actual cards, not just reference taxonomy
    # Group cards by their primary classification, exclude exact match with current
    sub_cls = {}
    for card in cards:
        cp = card.classification.primary
        if cp != udc_number:
            if cp not in sub_cls:
                sub_cls[cp] = ref.get_label(cp)
            sub_cls[cp] = ref.get_label(cp)
    children = [(cls_num, cls_label, sum(1 for c in cards if c.classification.primary == cls_num))
                for cls_num, cls_label in sorted(sub_cls.items())]
    ancestors = ref.get_ancestors(udc_number)
    return render_template(
        "browse.html",
        udc_number=udc_number,
        label=label,
        cards=cards,
        children=children,
        ancestors=ancestors,
        active_nav="catalog",
    )


@app.route("/search")
def search():
    q = request.args.get("q", "")
    cards = []
    if q:
        cards = search_cards(KB_PATH, q)
    return render_template("search.html", query=q, cards=cards, active_nav="search", search_query=q)


@app.route("/card/<card_id>")
def card_detail(card_id):
    card = read_card(KB_PATH, card_id)
    if not card:
        return "Card not found", 404
    return render_template("card.html", card=card, active_nav="catalog")


@app.route("/card/<card_id>/edit", methods=["GET", "POST"])
def card_edit(card_id):
    card = read_card(KB_PATH, card_id)
    if not card:
        return "Card not found", 404
    if request.method == "POST":
        card.title = request.form.get("title", card.title)
        card.abstract = request.form.get("abstract", card.abstract)
        card.content = request.form.get("content", card.content)
        card.author = request.form.get("author", card.author or "")
        card.source_url = request.form.get("source_url", card.source_url or "")
        card.tags = [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()]
        card.topics = [t.strip() for t in request.form.get("topics", "").split(",") if t.strip()]
        card.format = request.form.get("format", card.format or "")
        card.udc_label = request.form.get("udc_label", card.udc_label or "")
        # Parse sources from JSON textarea
        sources_json = request.form.get("sources_json", "").strip()
        if sources_json:
            try:
                card.sources = [Source.from_dict(json.loads(line)) for line in sources_json.split("\n") if line.strip()]
            except (json.JSONDecodeError, ValueError):
                pass  # Keep existing sources on parse error
        udc_number = request.form.get("udc_number", "").strip()
        if udc_number and udc_number != card.classification.primary:
            reclassify(KB_PATH, card_id, udc_number, ref)
            card = read_card(KB_PATH, card_id)
        else:
            write_card(card, KB_PATH, ref)
        invalidate_stats()
        return redirect(url_for("card_detail", card_id=card_id))
    return render_template("card_edit.html", card=card, active_nav="catalog")


@app.route("/ingest", methods=["GET", "POST"])
def ingest_page():
    result = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        source = request.form.get("source", "").strip()
        if source.startswith(("http://", "https://")):
            card = ingest_url(source, KB_PATH, ref)
            result = {"card": card}
        elif text:
            title = source or "Pasted Document"
            card = ingest(title, text, KB_PATH, ref, source_url=source)
            result = {"card": card}
        else:
            result = {"error": "Provide either document text or a URL to ingest"}
        invalidate_stats()
    return render_template("ingest.html", result=result, active_nav="ingest")


@app.route("/ingest/wikipedia_random", methods=["POST"])
def ingest_wikipedia_random_page():
    count = 10
    raw = request.form.get("count", "").strip()
    if raw:
        try:
            count = max(1, min(int(raw), 50))
        except ValueError:
            pass
    # Fetch random Wikipedia article titles via the API
    wp_resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "random", "rnnamespace": 0, "rnlimit": str(count), "format": "json"},
        headers={"User-Agent": "UniversalDecimalInator/1.0"},
        timeout=30,
    )
    wp_resp.raise_for_status()
    random_pages = wp_resp.json().get("query", {}).get("random", [])
    results = []
    for page in random_pages:
        title = page["title"]
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        try:
            card = ingest_url(url, KB_PATH, ref)
            results.append({"card": card, "error": None, "title": title})
        except Exception as e:
            logger.error("Failed to ingest Wikipedia article %s: %s", title, e)
            results.append({"card": None, "error": str(e), "title": title})
    invalidate_stats()
    return render_template(
        "ingest.html",
        result={"wikipedia_results": results},
        active_nav="ingest",
    )


@app.route("/card/<card_id>/delete", methods=["POST"])
def card_delete(card_id):
    delete_card(KB_PATH, card_id)
    invalidate_stats()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
