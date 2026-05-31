"""UniversalDecimalInator Card Catalog WebUI - Flask application."""

from __future__ import annotations

import logging
import os
import sys

import yaml
from flask import Flask, render_template, request, redirect, url_for, g

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniversalDecimalInator.reference import ClassificationReference
from UniversalDecimalInator.fs import (
    list_cards, search_cards, catalog_stats, read_card, write_card,
    cards_by_classification,
)
from UniversalDecimalInator.ingest import ingest_url
from UniversalDecimalInator.models import Card

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

KB_PATH = os.environ.get("KB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb"))
os.makedirs(KB_PATH, exist_ok=True)

REF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "udc_reference.yaml")
ref = ClassificationReference(REF_PATH)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.template_filter("udc_label")
def udc_label(num):
    return ref.get_label(str(num))


@app.before_request
def load_context():
    g.ref = ref
    g.stats = catalog_stats(KB_PATH)


@app.route("/")
def index():
    tree = ref.get_tree()
    main_classes = {k: v for k, v in ref.main_classes.items()}
    return render_template("index.html", tree=tree, main_classes=main_classes, stats=g.stats, ref=ref)


@app.route("/browse/<udc_number>")
def browse(udc_number):
    cards = cards_by_classification(KB_PATH, udc_number, ref)
    label = ref.get_label(udc_number)
    children = ref.get_children(udc_number)
    ancestors = ref.get_ancestors(udc_number)
    return render_template(
        "browse.html",
        udc_number=udc_number,
        label=label,
        cards=cards,
        children=children,
        ancestors=ancestors,
    )


@app.route("/search")
def search():
    q = request.args.get("q", "")
    cards = []
    if q:
        cards = search_cards(KB_PATH, q)
    return render_template("search.html", query=q, cards=cards)


@app.route("/card/<card_id>")
def card_detail(card_id):
    card = read_card(KB_PATH, card_id)
    if not card:
        return "Card not found", 404
    return render_template("card.html", card=card)


@app.route("/card/<card_id>/edit", methods=["GET", "POST"])
def card_edit(card_id):
    card = read_card(KB_PATH, card_id)
    if not card:
        return "Card not found", 404
    if request.method == "POST":
        card.abstract = request.form.get("abstract", card.abstract)
        card.tags = [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()]
        card.topics = [t.strip() for t in request.form.get("topics", "").split(",") if t.strip()]
        write_card(card, KB_PATH, ref)
        return redirect(url_for("card_detail", card_id=card_id))
    return render_template("card_edit.html", card=card)


@app.route("/ingest", methods=["GET", "POST"])
def ingest_page():
    result = None
    if request.method == "POST":
        source = request.form.get("source", "")
        if source.startswith(("http://", "https://")):
            result = ingest_url(source, KB_PATH, ref)
        else:
            result = {"error": "Provide a URL to ingest"}
    return render_template("ingest.html", result=result)


@app.route("/card/<card_id>/delete", methods=["POST"])
def card_delete(card_id):
    card = read_card(KB_PATH, card_id)
    if card and hasattr(card, 'path') and card.path:
        card.path.unlink()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
