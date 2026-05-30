# Dewey Decimal RAG System

## Design Principle

**All classification and summarization happens at ingest time, not query time.** Documents are processed once: summarized, classified into DDC categories, tagged, and embedded. At query time, the system only performs fast database filters and vector math — no LLM calls. This eliminates per-query latency and makes retrieval deterministic.

## Overview

A retrieval-augmented generation (RAG) system that uses Dewey Decimal Classification (DDC) numbers as a hierarchical routing layer to improve retrieval precision and reduce token waste. Documents are classified into DDC categories at ingest time, and queries are routed to relevant branches before vector search.

## Architecture

### Query Pipeline

```
Query → Metadata Filter (DDC + tags) → Abstract Match → Vector Search → Rerank → Return
```

1. **Metadata Filter**: Query is matched against pre-classified documents by DDC number and tags, narrowing the search space from the full corpus to a single branch (e.g., 100K docs → ~200 in 516.x). Uses database indexes — no LLM calls.
2. **Abstract Matching**: Lightweight similarity check against document summaries within the DDC branch (short text, fast comparison).
3. **Vector Search**: Embedding-based retrieval on the filtered subset for semantic matching.
4. **Rerank**: Cross-encoder reranker for final precision.
5. **Return**: Top K results with full text chunks.

**No LLM calls at query time.** All filtering uses pre-computed metadata and database indexes.

### Ingest Pipeline

```
Document → Summarize → Generate Metadata (DDC + tags + topics) → Embed → Store
```

Each document gets:
- **DDC Number**: Most specific classification (e.g., 516.37)
- **DDC Parent**: Parent class for fallback (e.g., 510)
- **Summary**: 50-100 word abstract
- **Tags**: Flat list of keywords (e.g., ["differential-geometry", "finsler-spaces"])
- **Topics**: Hierarchical subject areas (e.g., ["mathematics", "geometry"])
- **Audience**: Target audience level (e.g., "graduate", "academic")
- **Format**: Document type (e.g., "research-paper", "textbook")
- **Date**: Extracted publication date
- **Vector Embedding**: Semantic representation of the full text
- **Full Text**: Chunked and stored for retrieval

### Storage Schema

```yaml
document:
  id: string
  ddc_number: float          # e.g. 516.37
  ddc_parent: float          # e.g. 510 (for fallback)
  summary: string            # 50-100 word abstract
  tags: list[string]         # flat keywords
  topics: list[string]       # hierarchical subjects
  audience: string           # e.g. "graduate"
  format: string             # e.g. "research-paper"
  date: datetime             # extracted publication date
  embedding: vector          # document-level embedding
  chunks:
    - id: string
      text: string
      embedding: vector
      offset: int
```

### Index Strategy

Database indexes on: `ddc_number`, `tags`, `topics`, `audience`, `format`, `date`. All metadata filters use B-tree or inverted indexes for O(log n) lookups.

## Components

### 1. Document Summarizer

Generates abstracts for ingested documents.

**Input**: Full document text
**Output**: 50-100 word summary capturing core subject and key points

**Purpose**: Provides a compressed, noise-free representation for fast abstract matching at query time. Also used as input to the metadata generator (summarize first, then generate metadata from the summary to save tokens).

### 2. Metadata Generator

Generates all document metadata in a single LLM call at ingest time: DDC number, tags, topics, audience, format, and date extraction.

**Input**: Document summary (100 words)
**Output**: Structured metadata dict

**Rationale**: Amortizes LLM cost by generating all metadata in one call rather than separate calls for classification, tagging, etc.

**Prompt Template**:
```
Analyze this document summary and return structured metadata:

1. DDC Number: The most specific Dewey Decimal Classification number
2. Tags: 3-5 flat keywords (hyphenated, lowercase)
3. Topics: 2-3 hierarchical subject areas (broad to specific)
4. Audience: Target audience level (elementary, high-school, undergraduate, graduate, academic, general)
5. Format: Document type (research-paper, textbook, article, blog, report, etc.)
6. Date: Extracted publication date (or null if not present)

Main DDC classes:
000 - Computer science, information & general works
100 - Philosophy & psychology
200 - Religion
300 - Social sciences
400 - Language
500 - Pure science
600 - Technology
700 - Arts & recreation
800 - Literature
900 - History and geography

Summary: {summary}

Return as JSON.
```

### 3. Query Router

Routes incoming queries through the retrieval funnel using only pre-computed metadata.

**Flow**:
1. **Metadata filter**: Retrieve documents matching DDC branch + tags (database index lookup)
2. **Abstract match**: Score candidates by summary similarity (fast text comparison)
3. **Vector search**: Embedding-based retrieval on top abstract matches
4. **Rerank**: Cross-encoder reranker for final precision
5. **Return**: Top K results with full text chunks

**Fallback Strategy**:
- If metadata filter returns < N results: widen to parent DDC class
- If still < N results: expand to sibling DDC branches
- If still < N results: full corpus vector search with reranking

**No LLM calls at query time.** All filtering uses pre-computed metadata and database indexes.

### 4. Embedding Service

Generates vector embeddings for documents and queries.

**Requirements**:
- Supports both document-level and chunk-level embeddings
- Configurable embedding model (e.g., text-embedding-3-small, BGE, etc.)
- Runs at ingest time for documents, query time for queries

### 5. Reranker

Cross-encoder reranker for final precision after vector search.

**Requirements**:
- Configurable reranker model (e.g., BGE Reranker, Cohere Rerank)
- Takes query + candidate documents, returns ranked list

## Configuration

```yaml
# config.yaml
ddc:
  min_confidence: 0.7          # minimum confidence to use specific DDC; below this, use parent class
  fallback_parent: true        # fall back to parent class if confidence low
  expand_siblings: true        # search sibling branches if no results
  min_results_threshold: 10    # if fewer results than this, widen search

summarizer:
  model: string               # LLM model for summarization
  max_length: 100             # max words in summary
  temperature: 0.1

metadata_generator:
  model: string               # LLM model for metadata generation
  temperature: 0.1

embedding:
  model: string               # embedding model
  dimensions: int             # embedding dimensions

reranker:
  model: string               # cross-encoder reranker (e.g., BGE Reranker)

retrieval:
  top_k: 10                   # final results to return
  abstract_top_k: 20          # abstract matches to consider before vector search
  chunk_size: 500             # tokens per chunk
  chunk_overlap: 50           # overlap between chunks

storage:
  type: string                # "sqlite", "postgres", "chroma", etc.
  path: string                # database path or connection string
```

## API

### Ingest

```python
def ingest_document(text: str) -> Document:
    """Ingest a document: summarize, generate metadata, embed, and store."""
    summary = summarizer.summarize(text)
    metadata = metadata_generator.generate(summary)  # DDC, tags, topics, etc.
    embedding = embedder.embed(text)
    chunks = chunker.chunk(text)
    return storage.save(Document(text, summary, metadata, embedding, chunks))

def ingest_batch(documents: list[str]) -> list[Document]:
    """Ingest multiple documents in parallel."""
    return [ingest_document(doc) for doc in documents]
```

### Query

```python
def query(text: str, top_k: int = 10) -> list[Result]:
    """Search using metadata filters + vector search. No LLM calls."""
    candidates = storage.filter_by_metadata(text)  # DDC branch + tags
    abstract_matches = rank_by_abstract(text, candidates)
    vector_matches = rank_by_embedding(text, abstract_matches[:20])
    reranked = reranker.rerank(text, vector_matches)
    return reranked[:top_k]

def query_ddc(text: str, ddc_number: float, top_k: int = 10) -> list[Result]:
    """Search within a specific DDC branch."""
    candidates = storage.filter_by_ddc(ddc_number)
    return rank_by_embedding(text, candidates)[:top_k]
```

### Admin

```python
def get_ddc_distribution() -> dict[float, int]:
    """Return document count per DDC number."""
    return storage.ddc_distribution()

def reclassify(document_id: str, new_ddc: float) -> Document:
    """Manually override DDC classification."""
    return storage.update_ddc(document_id, new_ddc)
```

## DDC Main Classes Reference

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

## Multi-Classification Handling

Documents that span multiple DDC categories are tagged with all relevant numbers. The metadata generator returns a list of (ddc_number, confidence) pairs. All are stored and indexed — a document classified as both 170 and 006.7 will be found by queries matching either branch.

Example:
```yaml
document_id: doc_123
ddc_classifications:
  - number: 170    # moral philosophy
    confidence: 0.9
  - number: 006.7  # AI
    confidence: 0.85
  - number: 324.8  # civil liberties
    confidence: 0.7
```

## Consistency & Maintenance

- **Cluster analysis**: Periodically analyze documents in the same DDC branch for classification consistency
- **Normalization**: If >70% of documents in a cluster share a more specific number, standardize to that number
- **Gap detection**: Identify DDC branches with no documents to spot coverage gaps
- **Human review**: Flag low-confidence classifications (<0.5) for manual review
- **Batch re-ingest**: When taxonomy changes or quality degrades, re-run the ingest pipeline on affected documents

## Future Considerations

- **Custom DDC extensions**: Support for domain-specific subcategories beyond standard DDC
- **Cross-lingual**: DDC numbers are language-agnostic, enabling multilingual RAG
- **Batch re-ingest automation**: Detect classification drift and trigger re-ingest of affected documents
- **Trained classifier**: After 1,000+ LLM-labeled documents, train a lightweight classifier to replace the LLM for ingest
