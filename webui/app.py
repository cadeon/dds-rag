"""DDS-RAG Card Catalog WebUI - Flask application."""

from __future__ import annotations

import os
import sys

import yaml
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dds_rag.ddc import DDC_CLASSES, MAIN_CLASSES, get_ddc_tree, get_label, get_main_class
from dds_rag.storage import Storage
from dds_rag.admin import (
    catalog_stats, get_card, get_document, search_cards, list_cards_by_ddc,
    update_card_abstract, update_card_tags, update_card_topics,
    delete_card, reclassify,
)
from dds_rag.embedder import Embedder
from dds_rag.card_writer import CardWriter
from dds_rag.reranker import Reranker
from dds_rag.query_router import QueryRouter
from dds_rag.ingest import ingest

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize storage
db_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    config["storage"]["path"]
)
storage = Storage(db_path)

# Lazy-init LLM services (only when needed)
card_writer = None
embedder = None
reranker = None
query_router = None


def get_card_writer():
    global card_writer
    if not card_writer:
        cw = config["card_writer"]
        card_writer = CardWriter(
            model=cw["model"],
            endpoint=cw["endpoint"],
            temperature=cw.get("temperature", 0.1),
            max_abstract_length=cw.get("max_abstract_length", 100),
        )
    return card_writer


def get_embedder():
    global embedder
    if not embedder:
        emb = config["embedding"]
        embedder = Embedder(
            model=emb["model"],
            endpoint=emb.get("endpoint"),
            dimensions=emb.get("dimensions", 384),
        )
    return embedder


def get_reranker():
    global reranker
    if not reranker:
        rer = config["reranker"]
        reranker = Reranker(model=rer.get("model", "BAAI/bge-reranker-base"))
    return reranker


def get_query_router():
    global query_router
    if not query_router:
        dd = config["ddc"]
        query_router = QueryRouter(
            min_results_threshold=dd.get("min_results_threshold", 10),
            fallback_parent=dd.get("fallback_parent", True),
            expand_siblings=dd.get("expand_siblings", True),
            min_confidence=dd.get("min_confidence", 0.7),
        )
    return query_router


@app.route("/")
def index():
    """Main page - catalog overview with search."""
    stats = catalog_stats(storage)
    cards = storage.list_cards(limit=20)
    tree = get_ddc_tree()
    return render_template(
        "index.html",
        cards=cards,
        stats=stats["ddc_distribution"],
        total_cards=stats["total_cards"],
        total_documents=stats["total_documents"],
        tree=tree,
        main_classes=MAIN_CLASSES,
        ddc_classes=DDC_CLASSES,
    )


@app.route("/search")
def search():
    """Search page."""
    query = request.args.get("q", "")
    cards = []
    if query:
        cards = search_cards(query, storage)
    return render_template("search.html", cards=cards, query=query)


@app.route("/ddc/<float:ddc_number>")
def ddc_browse(ddc_number):
    """Browse cards by DDC classification."""
    cards = list_cards_by_ddc(ddc_number, storage)
    label = get_label(ddc_number)
    main = get_main_class(ddc_number)
    main_label = MAIN_CLASSES.get(main, "")
    return render_template(
        "ddc_browse.html",
        cards=cards,
        ddc_number=ddc_number,
        label=label,
        main=main,
        main_label=main_label,
    )


@app.route("/card/<card_id>")
def card_detail(card_id):
    """View a single card."""
    card = get_card(card_id, storage)
    if not card:
        return "Card not found", 404
    doc = get_document(card_id, storage)
    return render_template("card.html", card=card, document=doc, get_label=get_label)


@app.route("/card/<card_id>/edit", methods=["GET", "POST"])
def card_edit(card_id):
    """Edit a card's metadata."""
    card = get_card(card_id, storage)
    if not card:
        return "Card not found", 404

    if request.method == "POST":
        abstract = request.form.get("abstract", "")
        tags = [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()]
        topics = [t.strip() for t in request.form.get("topics", "").split(",") if t.strip()]
        new_ddc = request.form.get("ddc_number")

        if abstract:
            update_card_abstract(card_id, abstract, storage)
        if tags:
            update_card_tags(card_id, tags, storage)
        if topics:
            update_card_topics(card_id, topics, storage)
        if new_ddc:
            try:
                reclassify(card_id, float(new_ddc), storage)
            except ValueError:
                pass

        return redirect(url_for("card_detail", card_id=card_id))

    return render_template("card_edit.html", card=card, ddc_classes=DDC_CLASSES)


@app.route("/card/<card_id>/delete", methods=["POST"])
def card_delete(card_id):
    """Delete a card."""
    delete_card(card_id, storage)
    return redirect(url_for("index"))


@app.route("/ingest", methods=["GET", "POST"])
def ingest_page():
    """Ingest a new document."""
    if request.method == "POST":
        text = request.form.get("text", "")
        source = request.form.get("source", "")
        if text.strip():
            try:
                card = ingest(
                    text=text,
                    source=source,
                    card_writer=get_card_writer(),
                    embedder=get_embedder(),
                    storage=storage,
                    chunk_size=config["retrieval"]["chunk_size"],
                    chunk_overlap=config["retrieval"]["chunk_overlap"],
                )
                return redirect(url_for("card_detail", card_id=card.id))
            except Exception as e:
                return f"Ingest failed: {e}", 500
    return render_template("ingest.html")


@app.route("/query", methods=["POST"])
def query_api():
    """Query API endpoint."""
    data = request.json
    query_text = data.get("query", "")
    top_k = data.get("top_k", 10)
    ddc_number = data.get("ddc_number")

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    try:
        embedder = get_embedder()
        query_vec = embedder.embed(query_text)
        router = get_query_router()
        reranker = get_reranker()

        if ddc_number:
            chunks = router.route_ddc(
                query=query_text,
                query_vec=query_vec,
                ddc_number=float(ddc_number),
                storage=storage,
                reranker=reranker,
                top_k=top_k,
            )
        else:
            chunks = router.route(
                query=query_text,
                query_vec=query_vec,
                storage=storage,
                reranker=reranker,
                top_k=top_k,
            )

        results = []
        for chunk in chunks:
            results.append({
                "chunk_id": chunk.id,
                "text": chunk.text,
                "offset": chunk.offset,
            })

        return jsonify({"results": results, "count": len(results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """API: catalog statistics."""
    stats = catalog_stats(storage)
    return jsonify(stats)


@app.route("/api/cards", methods=["GET"])
def api_cards():
    """API: list cards with pagination."""
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    cards = storage.list_cards(limit=limit, offset=offset)
    return jsonify([c.to_dict() for c in cards])


@app.route("/api/cards/<card_id>", methods=["GET"])
def api_card(card_id):
    """API: get single card."""
    card = get_card(card_id, storage)
    if not card:
        return jsonify({"error": "Not found"}), 404
    return jsonify(card.to_dict())


@app.route("/api/cards/<card_id>", methods=["PUT"])
def api_card_update(card_id):
    """API: update card metadata."""
    data = request.json
    card = get_card(card_id, storage)
    if not card:
        return jsonify({"error": "Not found"}), 404

    if "abstract" in data:
        update_card_abstract(card_id, data["abstract"], storage)
    if "tags" in data:
        update_card_tags(card_id, data["tags"], storage)
    if "topics" in data:
        update_card_topics(card_id, data["topics"], storage)
    if "ddc_number" in data:
        reclassify(card_id, float(data["ddc_number"]), storage)

    card = get_card(card_id, storage)
    return jsonify(card.to_dict())


@app.route("/api/cards/<card_id>", methods=["DELETE"])
def api_card_delete(card_id):
    """API: delete card."""
    if delete_card(card_id, storage):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
