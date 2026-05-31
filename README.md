# UniversalDecimalInator

Universal Decimal Classification-based document catalog system.

Ingest documents, classify them with UDC compound numbers, and browse/search
a card catalog — no LLM calls at query time.

## Quick Start

```bash
pip install -e .
python -m webui.app
```

Browse to `http://localhost:5000`.

## Architecture

Documents are classified into UDC categories at ingest time using compound
notation (e.g., `004.738.5:179.4` for "machine learning in bioethics"). Cards
are stored as markdown files in a UDC-hierarchical directory structure.

## Configuration

Edit `config/udc_reference.yaml` to customize the classification hierarchy.
