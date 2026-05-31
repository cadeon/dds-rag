# UniversalDecimalInator Card Catalog

## Overview

A card catalog system using Universal Decimal Classification (UDC) for organizing and retrieving documents.

## Features

- UDC-based classification with compound notation support
- Card catalog interface with browse and search capabilities
- Ingest documents via URL or paste
- Edit and delete cards
- Full-text search across card content

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

```bash
python -m UniversalDecimalInator.ingest --url "https://example.com/article"
python -m UniversalDecimalInator.ingest --file "document.txt"
```

### Web UI

```bash
python -m webui.app
```

Browse to `http://localhost:5000` to access the card catalog.

## Configuration

Edit `config.yaml` for UDC classification settings and LLM endpoint configuration.

## License

CC BY-SA 4.0
