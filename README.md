# UniversalDecimalInator Card Catalog

## Overview

A card catalog system using Universal Decimal Classification (UDC) for organizing and retrieving documents. Documents are classified via LLM at ingest time — no embeddings or vector search.

## Features

- UDC-based classification with compound notation support
- Card catalog interface with browse and search capabilities
- Ingest documents via URL or paste through the web UI
- Edit and delete cards
- Full-text search across card content
- LLM-powered classification at ingest (no LLM needed at query time)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web UI

```bash
cd webui
python app.py
```

Browse to `http://localhost:5000` to access the card catalog.

## Configuration

Edit `config.yaml` for LLM endpoint and model configuration.
Edit `udc_reference.yaml` for UDC classification hierarchy.

## License

CC BY-SA 4.0
