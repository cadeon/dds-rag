# RAG-Friendly Output Structure — Discussion Notes

## Current State

Each card is a single `<card_id>.md` file with YAML frontmatter (metadata) and raw document text as body. Frontmatter has abstract, classification, tags, topics, source_url, author, timestamps.

### Problems

1. **No separation between metadata and searchable content** — RAG systems chunk the body but lose frontmatter context. A chunk about "neural network training" has no idea it came from a card classified as `006.2:519.8` with tags `[deep-learning, transformers]`. That metadata is valuable signal for hybrid search.

2. **Source documents are orphaned** — `kb/sources/<card_id>.txt` just stores the URL as a single line. If the user ingested a PDF, an image, or a local file, there's no place for it. The relationship between card and source is fragile.

3. **No structure within the content** — Dumping raw document text means the RAG chunks whatever it gets. If the document had sections, headings, or tables, that structure is preserved as plain text but the RAG has to rediscover it.

4. **Card and content in one file is both good and bad** — Good: atomic, one file per knowledge unit. Bad: the frontmatter is dead weight for chunking, and the abstract (which is a great summary) gets chunked alongside the full text, creating redundant low-value chunks.

---

## Approach A: Single file, enriched frontmatter, structured body

Keep one `.md` per card, but restructure:

```yaml
---
id: ...
title: ...
abstract: ...
classification:
  primary: ...
  secondary: [...]
tags: [...]
topics: [...]
source_url: ...
author: ...
created_at: ...
updated_at: ...
---

# SUMMARY
<abstract goes here as first section>

# SOURCE
<original document text, preserved with markdown structure>
```

The abstract becomes a named section in the body so RAG sees it as a distinct chunk. Source URL stays in frontmatter. Supporting files live in a `kb/<card_id>/` subdirectory.

**Pros:** One file per card, simple, most RAG ingestors handle markdown sections well. Metadata in frontmatter AND first section gives RAG context.

**Cons:** Still duplicates abstract (frontmatter + body). Not ideal if the RAG strips frontmatter entirely.

---

## Approach B: Card directory per knowledge unit

```
kb/<main>/<sub>/<udc>/<card_id>/
  card.md          # Frontmatter + abstract only (the "card" — metadata + summary)
  content.md       # Full original document text (the "content" — what gets chunked)
  sources/
    original.pdf   # Supporting/source files
    metadata.json  # Source provenance (URL, fetch date, content-type, etc.)
```

**Pros:** Clean separation. RAG ingests `content.md` for chunking, `card.md` for metadata enrichment. Source files co-located. No duplication.

**Cons:** More files, more complex directory traversal. RAG ingest needs to know the convention.

---

## Approach C: Single file with explicit RAG-friendly structure

```yaml
---
id: ...
title: ...
abstract: ...
classification:
  primary: ...
  secondary: [...]
tags: [...]
topics: [...]
source_url: ...
author: ...
created_at: ...
updated_at: ...
---

## Card: <title>
<abstract — 2-4 sentence summary>

## Classification
Primary: <UDC> | Secondary: <UDC, UDC> | Tags: <tag, tag, tag>

## Content
<original document text with preserved structure>

## Source
URL: <source_url>
```

Everything the RAG needs is in the body as named sections. Frontmatter is the machine-readable layer; the body is the human+RAG-readable layer. Classification and tags are repeated in the body so they survive chunking.

**Pros:** Maximum RAG compatibility — every chunk carries some context. Single file. No orphaned data.

**Cons:** More verbose. Some duplication (abstract in frontmatter and body, classification in both).

---

## Approach D: JSON + markdown sidecar

```
kb/<main>/<sub>/<udc>/
  <card_id>.json   # Machine-readable: full Card model as JSON
  <card_id>.md     # Human-readable: formatted card for browsing
  <card_id>-content.md  # Raw content for RAG chunking
```

**Pros:** Clean separation of concerns. JSON for programmatic access, markdown for humans, content file for RAG.

**Cons:** Three files per card. Sync burden on writes. Overkill for most RAG pipelines.

---

## What Matters Most

1. **The RAG needs the content to chunk** — non-negotiable. Whatever format, the full document text must be extractable.

2. **Metadata must travel with chunks** — classification, tags, and the abstract are the most valuable retrieval signals. If the RAG strips frontmatter, that context is lost.

3. **Source provenance should be preserved** — URL, fetch date, content type. Right now it's minimal.

4. **The abstract is gold** — it's an LLM-generated summary that's perfect for a "summary chunk" that ranks highly in semantic search without the noise of the full document.

## Preference

**Approach C** — single file, but the body explicitly repeats key metadata as named sections before the content. Most RAG chunkers respect markdown headings as chunk boundaries, so each chunk gets some context. Frontmatter stays for programmatic access. Simplest change with biggest RAG compatibility payoff.
