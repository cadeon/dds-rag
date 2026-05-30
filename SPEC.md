# DDS-RAG: Dewey Decimal System Retrieval Augmented Generation

## Problem Statement

Let's be honest: your organization's data is a mess. Documents are dumped into a vector store with no structure, no taxonomy, no hierarchy. When you query, the embedding model does its best, but it's searching through everything — marketing PDFs next to engineering docs next to HR policies — and semantically similar text doesn't always mean relevant text. Two documents can be "close" in embedding space but serve completely different purposes. Your RAG system retrieves what sounds right, not what belongs together.

Data needs to be organized to be useful. Classification isn't optional — it's the foundation of any retrieval system that actually works. Libraries figured this out in 1876. It's time RAG did too.

## Pitch

Augment your RAG with Double D's — Dewey Decimal System, not what you're thinking.

DDS-RAG adds a hierarchical classification layer to your retrieval pipeline. Documents are classified into Dewey Decimal categories at ingest time, so queries route to the right branch before they ever touch vector search. Think of it as giving your vector store a table of contents — and a card catalog. Every document gets a card: extracted metadata, an abstract, tags, classification. At query time, you browse the catalog first, then pull the actual text. Just like the good old days, except the librarians are LLMs and the shelves are vectors.

## Design Principle

**All classification and summarization happens at ingest time, not query time.** Documents are processed once: summarized, classified into DDC categories, tagged, and embedded. At query time, the system only performs fast database filters and vector math — no LLM calls. This eliminates per-query latency and makes retrieval deterministic.

## Architecture

### Query Pipeline

```
Query → Filter by DDC/Tags → Hybrid Search Cards → Reciprocal Rank Fusion → Pull Documents → Rerank → Return
```

1. **Filter by DDC/Tags**: Narrow the search space using card metadata. DDC branch is determined by lightweight keyword matching against the query (no LLM — just matching query terms against indexed card topics/tags to find the most relevant DDC branch). Reduces from the full corpus to a single branch (e.g., 100K cards → ~200 in 516.x). Uses database indexes.
2. **Hybrid search on cards**: Run **both** keyword (BM25) and vector similarity search against card abstracts in parallel. BM25 catches exact terms and technical jargon; vector search catches semantic meaning.
3. **Reciprocal Rank Fusion (RRF)**: Fuse the BM25 and vector result lists into a single ranked list of cards. Simple, parameter-free, and works well in practice.
4. **Pull documents**: Load the associated document chunks for the top fused cards.
5. **Rerank**: Cross-encoder reranker on the actual text chunks for final precision.
6. **Return**: Top K results with full text chunks.

**No LLM calls at query time.** All filtering and ranking uses pre-computed card metadata, keyword indexes, and vector embeddings.

### Ingest Pipeline

```
Document → Card Writer (summarize + metadata) → Embed → Store Card + Chunks
```

Each document produces a **Card** — the central catalog object:

- **DDC Number**: Most specific classification (e.g., 516.37)
- **DDC Parent**: Parent class for fallback (e.g., 510)
- **Abstract**: 50-100 word summary (the card's description)
- **Tags**: Flat list of keywords (e.g., ["differential-geometry", "finsler-spaces"])
- **Topics**: Hierarchical subject areas (e.g., ["mathematics", "geometry"])
- **Audience**: Target audience level (e.g., "graduate", "academic")
- **Format**: Document type (e.g., "research-paper", "textbook")
- **Date**: Extracted publication date
- **Embedding**: Vector representation of the abstract (not the full text)

The Card is what you search. The document text lives behind it, chunked and stored for retrieval once the right card is found.

### Storage Schema

```yaml
card:
  id: string
  ddc_classifications:        # multi-classification supported
    - number: float           # e.g. 516.37
      confidence: float       # e.g. 0.9
  ddc_parent: float           # parent class for fallback (e.g. 510)
  abstract: string            # 50-100 word summary
  tags: list[string]          # flat keywords
  topics: list[string]        # hierarchical subjects
  audience: string            # e.g. "graduate"
  format: string              # e.g. "research-paper"
  date: datetime              # extracted publication date
  embedding: vector           # embedding of the abstract

document:
  id: string
  card_id: string             # back-reference to the card
  source: string              # original filename/URL
  chunks:
    - id: string
      text: string
      embedding: vector
      offset: int
```

The card is the first-class citizen. Documents reference their card, not the other way around. A single card could eventually point to multiple documents (translations, revisions, related sources) — but for now it's one-to-one.

### Index Strategy

All indexes live on the **card** table: `ddc_classifications.number`, `tags`, `topics`, `audience`, `format`, `date`. Queries hit the card catalog first — document text is never scanned until a card is selected. All metadata filters use B-tree or inverted indexes for O(log n) lookups.

## Components

### 1. Card Writer

Creates catalog cards from raw documents. Combines summarization and metadata generation into a single pipeline.

**Pipeline**:
1. **Summarize and classify**: Generate a 50-100 word abstract and extract all metadata (DDC classifications, tags, topics, audience, format, date) from the full document text in a single LLM call.

**Input**: Full document text
**Output**: A complete Card (abstract + all metadata fields)

**Rationale**: Amortizes LLM cost by generating the abstract and all metadata in a single call. The abstract is the card's description — short enough to embed cheaply, rich enough to distinguish this document from others on the same shelf.

**Prompt Template**: See [templates/card_writer.md](templates/card_writer.md) for the full prompt. Key points:
- Asks for abstract, DDC classifications, tags, topics, audience, format, date
- Includes full DDC hierarchy down to 100-level (see DDC Reference below)
- Enforces hyphenated lowercase tags
- Returns JSON

### 2. Query Router

Routes incoming queries through the card catalog using hybrid search — keyword and vector in parallel, fused with reciprocal rank fusion.

**Why hybrid?**: BM25 catches exact terms and technical jargon; vector search catches semantic meaning. Cards are short (50-100 word abstracts), so both searches are fast. RRF merges results: `score = sum(1 / (k + rank))` where k=60.

**Fallback Strategy**:
- If catalog returns < N cards: widen to parent DDC class
- If still < N: expand to sibling DDC branches
- If still < N: full corpus hybrid search with reranking

**No LLM calls at query time.**

### 3. Embedding Service

Embeds card abstracts, document chunks, and queries. Configurable model (e.g., text-embedding-3-small, BGE). All embeddings computed at ingest except query embeddings.

### 4. Reranker

Cross-encoder reranker on final text chunks. Configurable model (e.g., BGE Reranker, Cohere Rerank).

## Configuration

```yaml
# config.yaml
ddc:
  min_confidence: 0.7          # minimum confidence to use specific DDC; below this, use parent class
  fallback_parent: true        # fall back to parent class if confidence low
  expand_siblings: true        # search sibling branches if no results
  min_results_threshold: 10    # if fewer results than this, widen search

card_writer:
  model: string               # LLM model for card creation
  max_abstract_length: 100    # max words in abstract
  temperature: 0.1

embedding:
  model: string               # embedding model
  dimensions: int             # embedding dimensions

reranker:
  model: string               # cross-encoder reranker (e.g., BGE Reranker)

retrieval:
  top_k: 10                   # final results to return
  fused_top_k: 20             # cards to consider after RRF before pulling chunks
  chunk_size: 500             # tokens per chunk
  chunk_overlap: 50           # overlap between chunks

storage:
  type: string                # "sqlite", "postgres", "chroma", etc.
  path: string                # database path or connection string
```

## API

### Ingest

```python
def ingest(text: str, source: str = "") -> Card:
    """Create a catalog card from raw document text."""
    card = card_writer.write(text)           # abstract + metadata
    card.embedding = embedder.embed(card.abstract)
    chunks = chunker.chunk(text)
    for chunk in chunks:
        chunk.embedding = embedder.embed(chunk.text)
    document = Document(card_id=card.id, source=source, chunks=chunks)
    storage.save_card(card)
    storage.save_document(document)
    return card

def ingest_batch(documents: list[str], sources: list[str] = None) -> list[Card]:
    """Ingest multiple documents in parallel."""
    return [ingest(doc, src) for doc, src in zip(documents, sources or [""]*len(documents))]
```

### Query

```python
def query(text: str, top_k: int = 10) -> list[Result]:
    """Hybrid search the card catalog, pull documents, return ranked chunks."""
    query_vec = embedder.embed(text)
    candidate_cards = storage.browse_catalog(text)   # keyword match topics/tags → DDC branch
    bm25_cards = storage.bm25_search(text, candidate_cards)
    vector_cards = storage.vector_search(query_vec, candidate_cards)
    fused_cards = reciprocal_rank_fusion(bm25_cards, vector_cards)
    chunks = storage.pull_chunks([c.id for c in fused_cards[:20]])
    reranked = reranker.rerank(text, chunks)
    return reranked[:top_k]

def query_ddc(text: str, ddc_number: float, top_k: int = 10) -> list[Result]:
    """Hybrid search within a specific DDC branch."""
    cards = storage.cards_by_ddc(ddc_number)
    bm25_cards = storage.bm25_search(text, cards)
    vector_cards = storage.vector_search(embedder.embed(text), cards)
    fused_cards = reciprocal_rank_fusion(bm25_cards, vector_cards)
    chunks = storage.pull_chunks([c.id for c in fused_cards])
    return chunks[:top_k]
```

### Admin

```python
def catalog_stats() -> dict[float, int]:
    """Return card count per DDC number."""
    return storage.ddc_distribution()

def reclassify(card_id: str, new_ddc: float) -> Card:
    """Manually override card classification."""
    return storage.update_ddc(card_id, new_ddc)

def get_card(card_id: str) -> Card:
    """Look up a card by ID."""
    return storage.get_card(card_id)

def get_document(card_id: str) -> Document:
    """Pull the full document behind a card."""
    return storage.get_document_by_card(card_id)
```

## DDC Classification

DDC numbers are the backbone. Getting them right at ingest determines whether queries find what they're looking for.

### Main Classes

| Number | Class |
|--------|-------|
| 000 | Computer science, information & general works |
| 100 | Philosophy & psychology |
| 200 | Religion |
| 300 | Social sciences |
| 400 | Language |
| 500 | Pure science |
| 600 | Technology |
| 700 | Arts & recreation |
| 800 | Literature |
| 900 | History and geography |

### Classification Strategy

The Card Writer assigns DDC numbers from the full DDC table (not just the 10 main classes). The prompt includes the complete hierarchy down to the 100-level (e.g., 510 → 516 → 516.3 → 516.36) so the LLM picks specific numbers, not broad categories. A document about differential geometry should be 516.36, not 516, not 510.

### Matching to Existing Classifications

When the catalog already has cards, prefer reusing existing DDC numbers over inventing new ones. If 200 cards are already 516.36, a new differential geometry paper should also be 516.36.

**Implementation**: At ingest, query the catalog for the most common DDC numbers in the relevant branch. Pass the top N to the Card Writer prompt as "preferred classifications" — the LLM can override but is biased toward reuse.

### New Classifications

DDC has over 40,000 numbers. The LLM may assign numbers no other card uses — that's fine. A new number just means a new shelf. Consistent reuse of a "new" number means the taxonomy is working; scattering across numbers means a consistency problem caught by maintenance.

### Keyword Normalization

Tags and topics: lowercase, hyphenated, stemmed (Snowball). No custom synonym resolution — BM25 handles token matching. Prompt enforces format at source.

### Multi-Classification

Cards spanning multiple DDC categories carry all relevant numbers as (number, confidence) pairs. All are indexed — a card classified as both 170 and 006.7 is found by queries matching either branch.

### Consistency & Maintenance

- **Cluster analysis**: Periodically check cards in the same DDC branch for consistency
- **Normalization**: If >70% of a cluster shares a more specific number, standardize
- **Gap detection**: Identify empty DDC branches
- **Human review**: Flag low-confidence classifications (<0.5)
- **Batch re-ingest**: Re-run Card Writer when taxonomy changes or quality degrades

## Future Considerations

- Custom DDC extensions for domain-specific subcategories
- Cross-lingual RAG (DDC numbers are language-agnostic)
- Automated classification drift detection
- Trained classifier to replace LLM after 1,000+ labeled cards
