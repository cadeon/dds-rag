# UniversalDecimalInator Card Catalog

Universal Decimal Classification-based document catalog system.

Ingest documents, classify them with UDC compound numbers, and browse/search a
card catalog. All classification and metadata generation happens at **ingest
time** — no LLM calls at query time.

## Features

- UDC-based classification with compound notation support
- Card catalog interface with browse and search capabilities
- Ingest documents via URL or paste through the web UI
- Edit and delete cards
- Full-text search across card content
- LLM-powered classification at ingest (no LLM needed at query time)

## Architecture

Documents are classified into UDC categories at ingest time using compound
notation (e.g., `004.738.5:179.4` for "machine learning in bioethics"). Cards
are stored as markdown files in a UDC-hierarchical directory structure.

### Key Components

- **ClassificationReference** (`UniversalDecimalInator/reference.py`): Loads taxonomy from YAML
- **CardWriter** (`UniversalDecimalInator/card_writer.py`): LLM-powered classification
- **Filesystem** (`UniversalDecimalInator/fs.py`): Card storage and retrieval
- **WebUI** (`webui/app.py`): Flask-based browse/search interface

### Directory Structure

```
kb/
  0/          # Computer science, information
    004/
      004.738.5/
        card.md
  5/          # Natural sciences, mathematics
    519/
      519.684/
        card.md
  sources/    # Source URLs (one per card, stored as {card_id}.txt)
```

### Card Format

Each card is a markdown file with YAML frontmatter:

```markdown
---
id: my-card-id
title: "My Document"
abstract: "Brief summary..."
primary_classification: "004.738.5"
secondary_classifications: []
tags: ["tag1", "tag2"]
topics: ["Topic Area"]
source_url: ""
author: ""
---

Document content goes here...
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web UI

```bash
cd webui && python app.py
```

Browse to `http://localhost:5000`.

### Python API

```python
from UniversalDecimalInator import ingest, list_cards, search_cards, ClassificationReference

ref = ClassificationReference("udc_reference.yaml")
card = ingest("Title", "Content...", "kb", ref)
results = search_cards("kb", "machine learning")
```

## Configuration

Edit `config.yaml` for LLM endpoint and model settings.
Edit `udc_reference.yaml` to customize the classification hierarchy.

## License

CC BY-SA 4.0
