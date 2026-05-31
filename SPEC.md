# UniversalDecimalInator

Universal Decimal Classification-based document catalog system.

## Overview

Ingest documents, classify them with UDC compound numbers, and browse/search a
card catalog. All classification and metadata generation happens at **ingest
time** — no LLM calls at query time.

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
  sources/    # Original document text
```

## Configuration

Edit `config/udc_reference.yaml` to customize the classification hierarchy.

## Quick Start

```bash
pip install -e .
python -m webui.app
```

Browse to `http://localhost:5000`.
