"""Seed the DDS-RAG card catalog with sample data.

Usage:
    python scripts/seed_catalog.py          # seed the default DB
    python scripts/seed_catalog.py path.db   # seed a specific DB
    python scripts/seed_catalog.py --clear   # clear existing data first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dds_rag.models import Card, Chunk, Document, DDCClassification
from dds_rag.storage import Storage

# ---------------------------------------------------------------------------
# Sample documents — each entry: (card_id, ddc, ddc_parent, abstract, tags,
# topics, audience, fmt, date, source_url, chunks)
# ---------------------------------------------------------------------------

SAMPLES = [
    # 000 — Computer science
    {
        "id": "seed-001",
        "ddc": [3, 0.95],
        "title": "Comprehensive Guide to Relational Database Design",
        "abstract": "Comprehensive guide to relational database design covering normalization, indexing strategies, query optimization, and ACID transactions for building robust data storage systems.",
        "author": "Database Systems Institute",
        "tags": ["database", "SQL", "relational", "data-modeling"],
        "topics": ["database design", "normalization", "query optimization", "transactions"],
        "audience": "developers",
        "format": "book",
        "date": "2024-03-15",
        "source": "https://example.com/database-design",
        "chunks": [
            "Relational databases organize data into tables with rows and columns. Each table represents an entity, and relationships between tables are established through foreign keys. Proper normalization eliminates redundancy while maintaining data integrity.",
            "Indexing is critical for query performance. B-tree indexes support equality and range queries, while hash indexes excel at exact-match lookups. Composite indexes should be designed based on common query patterns, placing the most selective columns first.",
            "ACID properties ensure reliable transactions. Atomicity guarantees all-or-nothing execution. Consistency maintains database invariants. Isolation prevents concurrent transactions from interfering. Durability ensures committed data survives system failures.",
        ],
    },
    {
        "id": "seed-002",
        "ddc": [5, 0.92],
        "title": "Python Programming Fundamentals: Data Structures, Algorithms, and Modern Features",
        "abstract": "Python programming fundamentals covering data structures, algorithms, object-oriented design, and modern Python features including type hints, async/await, and context managers.",
        "author": "Python Education Foundation",
        "tags": ["python", "programming", "algorithms", "OOP"],
        "topics": ["Python", "data structures", "algorithms", "object-oriented programming"],
        "audience": "beginners",
        "format": "tutorial",
        "date": "2024-06-20",
        "source": "https://example.com/python-guide",
        "chunks": [
            "Python is a dynamically typed, interpreted language known for readability. Its built-in data structures — lists, dictionaries, sets, and tuples — handle most common programming tasks without external dependencies.",
            "Object-oriented programming in Python uses classes and inheritance. Python supports multiple inheritance, mixins, and duck typing. The MRO (Method Resolution Order) follows the C3 linearization algorithm.",
            "Async/await enables concurrent I/O operations without threads. The asyncio event loop manages coroutines, allowing thousands of concurrent connections. Use async for network-bound tasks, not CPU-bound work.",
        ],
    },
    {
        "id": "seed-003",
        "ddc": [7, 0.88],
        "title": "Modern Web Application Architecture: Microservices, Containers, and Event-Driven Systems",
        "abstract": "Modern web application architecture using microservices, containerization with Docker and Kubernetes, API gateway patterns, and event-driven communication between distributed services.",
        "author": "Cloud Architecture Review",
        "tags": ["microservices", "Docker", "Kubernetes", "distributed-systems"],
        "topics": ["microservices", "containerization", "API gateway", "event-driven architecture"],
        "audience": "architects",
        "format": "article",
        "date": "2025-01-10",
        "source": "https://example.com/microservices-arch",
        "chunks": [
            "Microservices decompose monolithic applications into independently deployable services. Each service owns its data, communicates via APIs, and can scale independently. The tradeoff is operational complexity.",
            "Docker containers package applications with dependencies. Kubernetes orchestrates containers across clusters, handling scheduling, scaling, and self-healing. Services, Deployments, and ConfigMaps are core primitives.",
            "Event-driven architecture uses message brokers like Kafka or RabbitMQ. Services publish events and subscribe to topics, achieving loose coupling. Saga patterns handle distributed transactions across services.",
        ],
    },
    # 100 — Philosophy & psychology
    {
        "id": "seed-101",
        "ddc": [153, 0.91],
        "title": "Human Memory Systems: Working Memory, Consolidation, and Attention",
        "abstract": "Cognitive psychology research on human memory systems, including working memory limits, long-term memory consolidation, and the role of attention in encoding and retrieval.",
        "author": "Journal of Cognitive Psychology",
        "tags": ["psychology", "memory", "cognition", "neuroscience"],
        "topics": ["working memory", "long-term memory", "attention", "cognitive load"],
        "audience": "students",
        "format": "paper",
        "date": "2024-09-05",
        "source": "https://example.com/memory-research",
        "chunks": [
            "Working memory holds approximately 7 plus or minus 2 items simultaneously. Miller's classic finding has been refined to suggest 4 chunks for most adults. Cognitive load theory applies this to instructional design.",
            "Long-term memory consolidates through sleep-dependent processes. The hippocampus replays daily experiences during slow-wave sleep, transferring information to neocortical storage. Spaced repetition exploits this mechanism.",
            "Attention acts as a bottleneck between sensory input and memory encoding. Divided attention during learning significantly impairs retention. The attentional blink phenomenon shows we miss stimuli within 200-500ms of detecting another.",
        ],
    },
    {
        "id": "seed-102",
        "ddc": [170, 0.87],
        "title": "Ethical Frameworks: From Utilitarianism to AI Ethics",
        "abstract": "Introduction to ethical frameworks including utilitarianism, deontology, virtue ethics, and care ethics, with applications to contemporary issues in technology and artificial intelligence.",
        "author": "Philosophy Press",
        "tags": ["ethics", "philosophy", "AI-ethics", "moral-reasoning"],
        "topics": ["utilitarianism", "deontology", "virtue ethics", "applied ethics"],
        "audience": "general",
        "format": "book",
        "date": "2024-11-12",
        "source": "https://example.com/ethics-intro",
        "chunks": [
            "Utilitarianism evaluates actions by consequences, maximizing overall happiness. Bentham's quantitative approach counts pleasures, while Mill's qualitative version distinguishes higher and lower pleasures. Rule utilitarianism avoids case-by-case calculations.",
            "Deontological ethics, associated with Kant, judges actions by adherence to moral rules regardless of outcomes. The categorical imperative demands acting only on maxims that could be universal laws. Duties are absolute, not contingent on results.",
            "Virtue ethics focuses on character rather than rules or consequences. Aristotle's golden mean finds virtue between excess and deficiency. Courage lies between cowardice and recklessness. Modern virtue ethicists emphasize moral exemplars and practical wisdom.",
        ],
    },
    # 300 — Social sciences
    {
        "id": "seed-301",
        "ddc": [303.4833, 0.94],
        "title": "Social Media's Impact on Democratic Processes: Echo Chambers, Misinformation, and Governance",
        "abstract": "Analysis of social media's impact on democratic processes, including algorithmic echo chambers, misinformation spread, voter manipulation, and platform governance models.",
        "author": "Digital Democracy Research Center",
        "tags": ["social-media", "democracy", "misinformation", "governance"],
        "topics": ["social media", "echo chambers", "misinformation", "digital democracy"],
        "audience": "policymakers",
        "format": "report",
        "date": "2025-02-28",
        "source": "https://example.com/social-media-democracy",
        "chunks": [
            "Social media algorithms optimize for engagement, amplifying emotionally charged content. This creates filter bubbles where users encounter only confirming viewpoints. Studies show polarization increases by 23% among heavy social media users.",
            "Misinformation spreads six times faster than factual content on Twitter. Bots amplify false claims during elections, while deepfakes create synthetic media indistinguishable from reality. Platform fact-checking has mixed effectiveness.",
            "Platform governance models range from content moderation to cryptographic verification. The EU Digital Services Act mandates transparency in recommendation algorithms. Decentralized protocols like ActivityPub offer alternatives to centralized control.",
        ],
    },
    {
        "id": "seed-302",
        "ddc": [330.9, 0.89],
        "title": "Economic Analysis of Remote Work: Productivity, Urban Real Estate, and Labor Markets",
        "abstract": "Economic analysis of remote work trends, examining productivity metrics, urban real estate impacts, commuting patterns, and the redistribution of economic activity from cities to suburbs and rural areas.",
        "author": "Economic Policy Institute",
        "tags": ["economics", "remote-work", "urban-planning", "labor"],
        "topics": ["remote work", "productivity", "urban economics", "labor markets"],
        "audience": "econom",
        "format": "analysis",
        "date": "2025-04-01",
        "source": "https://example.com/remote-work-economics",
        "chunks": [
            "Remote work productivity studies show a 13% average increase, though collaboration-heavy tasks suffer. The hybrid model — three days office, two remote — dominates Fortune 500 policies. Output-based management replaces presenteeism.",
            "Commercial real estate vacancy rates in major cities reached 22% in 2024. Office-to-residential conversions face zoning and structural challenges. Suburban property values increased 15% while downtown commercial declined 30%.",
            "Commuting redistribution saves workers 8 billion hours annually in the US alone. However, reduced transit ridership threatens public transportation funding. Cities are redesigning downtowns for mixed-use rather than office-dominated development.",
        ],
    },
    # 500 — Pure science
    {
        "id": "seed-501",
        "ddc": [512.8, 0.93],
        "title": "Machine Learning Foundations: Neural Networks, Optimization, and Regularization",
        "abstract": "Machine learning foundations covering supervised and unsupervised learning, neural network architectures, gradient descent optimization, and regularization techniques for preventing overfitting.",
        "author": "ML Education Lab",
        "tags": ["machine-learning", "neural-networks", "optimization", "statistics"],
        "topics": ["supervised learning", "neural networks", "gradient descent", "regularization"],
        "audience": "developers",
        "format": "course",
        "date": "2024-07-18",
        "source": "https://example.com/ml-foundations",
        "chunks": [
            "Supervised learning maps inputs to known outputs using labeled training data. Classification predicts discrete categories while regression predicts continuous values. Cross-validation estimates generalization performance on unseen data.",
            "Neural networks consist of layers of interconnected neurons. Deep learning uses multiple hidden layers to learn hierarchical representations. Backpropagation computes gradients efficiently using the chain rule of calculus.",
            "Gradient descent iteratively adjusts weights to minimize loss. Stochastic variants — SGD, Adam, RMSProp — add momentum and adaptive learning rates. Regularization techniques like L2 penalties and dropout prevent overfitting to training data.",
        ],
    },
    {
        "id": "seed-502",
        "ddc": [576.869, 0.86],
        "title": "CRISPR-Cas9 Gene Editing: Molecular Biology, Off-Target Effects, and Therapeutic Applications",
        "abstract": "Molecular biology of CRISPR-Cas9 gene editing, including guide RNA design, off-target effects, delivery mechanisms, and therapeutic applications for genetic diseases.",
        "author": "Molecular Biology Review Journal",
        "tags": ["CRISPR", "gene-editing", "molecular-biology", "genetics"],
        "topics": ["CRISPR-Cas9", "gene therapy", "guide RNA", "off-target effects"],
        "audience": "researchers",
        "format": "review",
        "date": "2025-03-22",
        "source": "https://example.com/crispr-review",
        "chunks": [
            "CRISPR-Cas9 uses a guide RNA to direct the Cas9 nuclease to specific DNA sequences. The PAM sequence (NGG for SpCas9) determines targetability. Double-strand breaks are repaired by non-homologous end joining or homology-directed repair.",
            "Off-target effects remain the primary safety concern. High-fidelity Cas9 variants reduce off-targets by 100x. Whole-genome sequencing and GUIDE-seq detect unintended modifications. Base editing avoids double-strand breaks entirely.",
            "Therapeutic CRISPR treatments include ex vivo editing of hematopoietic stem cells for sickle cell disease and in vivo liver editing for hereditary transthyretin amyloidosis. Lipid nanoparticles and AAV vectors deliver CRISPR components in vivo.",
        ],
    },
    # 600 — Technology
    {
        "id": "seed-601",
        "ddc": [621.384, 0.90],
        "title": "Renewable Energy Systems Integration: Solar, Wind, Storage, and Smart Grids",
        "abstract": "Renewable energy systems integration covering solar PV, wind turbines, battery storage, smart grid technologies, and the technical challenges of high-penetration renewable electricity grids.",
        "author": "Energy Engineering Handbook",
        "tags": ["renewable-energy", "solar", "wind", "grid", "battery-storage"],
        "topics": ["solar PV", "wind energy", "energy storage", "smart grid"],
        "audience": "engineers",
        "format": "handbook",
        "date": "2024-12-01",
        "source": "https://example.com/renewable-energy",
        "chunks": [
            "Solar PV efficiency has reached 26.7% for commercial modules. Perovskite-silicon tandem cells exceed 33% in lab settings. Balance-of-system costs — inverters, mounting, wiring — account for 40% of total installation expenses.",
            "Wind turbine capacity factors average 35% onshore and 50% offshore. Variable-speed, pitch-controlled turbines maximize energy capture across wind speeds. Wake effects reduce downstream turbine output by 10-20% in dense arrays.",
            "Lithium-ion battery storage costs fell 89% since 2010. Grid-scale batteries provide frequency regulation, energy arbitrage, and peak shaving. Emerging technologies include flow batteries for long-duration storage and solid-state cells for safety.",
        ],
    },
    {
        "id": "seed-602",
        "ddc": [628.06, 0.85],
        "title": "Cybersecurity Fundamentals: Network Defense, Encryption, and Zero-Trust Architecture",
        "abstract": "Cybersecurity fundamentals: network defense, encryption protocols, threat modeling, incident response procedures, and zero-trust architecture for protecting organizational infrastructure.",
        "author": "Cybersecurity Standards Board",
        "tags": ["cybersecurity", "encryption", "network-security", "zero-trust"],
        "topics": ["threat modeling", "incident response", "encryption", "zero-trust architecture"],
        "audience": "IT-professionals",
        "format": "guide",
        "date": "2025-05-15",
        "source": "https://example.com/cybersecurity-fundamentals",
        "chunks": [
            "Zero-trust architecture assumes no implicit trust. Every request is authenticated, authorized, and encrypted. Micro-segmentation isolates workloads. Identity becomes the new perimeter, with continuous verification replacing static firewalls.",
            "Encryption at rest uses AES-256, while TLS 1.3 protects data in transit. Post-quantum cryptography — lattice-based and code-based algorithms — prepares for quantum computer threats. Key management is the weakest link in any encryption strategy.",
            "Incident response follows six phases: preparation, identification, containment, eradication, recovery, and lessons learned. Mean time to detect breaches has decreased to 207 days, while mean time to contain remains 72 days.",
        ],
    },
    # 700 — Arts & recreation
    {
        "id": "seed-701",
        "ddc": [781.6, 0.88],
        "title": "The Evolution of Electronic Music Production: Synthesizers, DAWs, and Digital Culture",
        "abstract": "The evolution of electronic music production from analog synthesizers to digital audio workstations, covering sampling, MIDI, effects processing, and the cultural impact of genres from house to dubstep.",
        "author": "Electronic Music Archive",
        "tags": ["electronic-music", "synthesizers", "DAW", "music-production"],
        "topics": ["synthesizers", "MIDI", "music production", "electronic genres"],
        "audience": "musicians",
        "format": "book",
        "date": "2024-08-30",
        "source": "https://example.com/electronic-music",
        "chunks": [
            "Analog synthesizers generate sound through oscillators, filters, and amplifiers controlled by envelope generators. Subtractive synthesis starts with rich waveforms and removes frequencies. The Moog, Minimoog, and Prophet-5 defined early electronic music tones.",
            "MIDI (Musical Instrument Digital Interface) standardizes communication between electronic instruments. It transmits note events, not audio. Digital audio workstations like Ableton Live and Logic Pro combine sequencing, recording, and effects in software.",
            "Sampling revolutionized music production. The MPC 3000 and Akai S-series samplers enabled hip-hop producers to manipulate recorded sounds. Time-stretching and pitch-shifting algorithms allow samples to fit any tempo and key.",
        ],
    },
    # 800 — Literature
    {
        "id": "seed-801",
        "ddc": [823.914, 0.91],
        "title": "Postcolonial Literature: Narrative Authority, Hybridity, and Decolonization",
        "abstract": "Critical analysis of postcolonial literature, examining how authors from formerly colonized regions reclaim narrative authority, subvert colonial discourse, and construct hybrid cultural identities through fiction.",
        "author": "Literary Studies Quarterly",
        "tags": ["postcolonial", "literature", "cultural-identity", "fiction"],
        "topics": ["postcolonialism", "narrative authority", "hybridity", "decolonization"],
        "audience": "students",
        "format": "essay",
        "date": "2024-10-08",
        "source": "https://example.com/postcolonial-lit",
        "chunks": [
            "Postcolonial literature emerged as formerly colonized nations gained independence. Authors like Chinua Achebe, Ngugi wa Thiong'o, and Salman Rushdie wrote in and about colonial languages while subverting their imperial associations.",
            "Homi Bhabha's concept of hybridity describes the cultural mixing that occurs in colonial encounters. Postcolonial texts occupy the 'third space' between colonizer and colonized, creating new forms of expression that transcend binary oppositions.",
            "Narrative authority is central to postcolonial fiction. Achebe's Things Fall Apart retells African history from an Igbo perspective, countering Conrad's Heart of Darkness. Magical realism blends indigenous worldviews with Western literary forms.",
        ],
    },
    # 900 — History and geography
    {
        "id": "seed-901",
        "ddc": [940.53, 0.92],
        "title": "The Fall of the Berlin Wall and German Reunification: A Historical Analysis",
        "abstract": "The fall of the Berlin Wall and German reunification: diplomatic negotiations, economic integration challenges, social divisions between East and West Germans, and the lasting political consequences for European unity.",
        "author": "European History Institute",
        "tags": ["Cold-War", "Germany", "reunification", "European-history"],
        "topics": ["Berlin Wall", "German reunification", "Cold War", "European integration"],
        "audience": "general",
        "format": "documentary",
        "date": "2024-04-25",
        "source": "https://example.com/berlin-wall",
        "chunks": [
            "The Berlin Wall fell on November 9, 1989, after a miscommunication at a press conference led to its opening. Two million East Germans had emigrated since 1961, primarily through Hungary's briefly opened border in May.",
            "Reunification on October 3, 1990, came through the Two Plus Four Treaty. The East German mark was converted at 1:1 for wages up to 4000 marks. Treuhand privatized 8,500 East German enterprises, closing 70% within three years.",
            "East-West divisions persist. Eastern Germany votes differently, has lower wages, and higher unemployment. The 'Ossi-Kessi' divide shapes German politics. Yet reunification strengthened the EU, with Poland and Czech Republic joining in 2004.",
        ],
    },
]


def seed(db_path: str, clear: bool = False) -> None:
    """Populate the card catalog with sample data."""
    storage = Storage(db_path)

    if clear:
        # Delete all existing cards (cascades to documents/chunks)
        conn = storage._conn
        try:
            rows = conn.execute("SELECT id FROM cards").fetchall()
            for row in rows:
                storage.delete_card(row["id"])
            print(f"Cleared {len(rows)} existing cards.")
        finally:
            conn.close()

    count = 0
    for sample in SAMPLES:
        card_id = sample["id"]
        ddc_num = sample["ddc"][0]
        ddc_conf = sample["ddc"][1]

        # Determine parent DDC
        parent = float(int(ddc_num // 100) * 100)

        card = Card(
            id=card_id,
            ddc_classifications=[DDCClassification(number=ddc_num, confidence=ddc_conf)],
            ddc_parent=parent,
            title=sample.get("title", ""),
            abstract=sample["abstract"],
            author=sample.get("author", "Unknown"),
            source_url=sample.get("source", ""),
            tags=sample["tags"],
            topics=sample["topics"],
            audience=sample["audience"],
            format=sample["format"],
            date=sample["date"],
        )
        storage.save_card(card)

        doc_id = f"doc-{card_id}"
        chunks = [
            Chunk(
                id=f"chunk-{card_id}-{i}",
                text=text,
                offset=i * 500,
            )
            for i, text in enumerate(sample["chunks"])
        ]
        doc = Document(
            id=doc_id,
            card_id=card_id,
            source=sample["source"],
            chunks=chunks,
        )
        storage.save_document(doc)
        count += 1

    total_cards = storage.total_cards()
    total_docs = storage.total_documents()
    dist = storage.ddc_distribution()

    print(f"Seeded {count} cards with documents.")
    print(f"Total cards: {total_cards}, Total documents: {total_docs}")
    print(f"DDC distribution: {json.dumps(dist, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Seed the DDS-RAG card catalog")
    parser.add_argument("db_path", nargs="?", default=None, help="Path to SQLite DB")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")
    args = parser.parse_args()

    if args.db_path:
        db_path = args.db_path
    else:
        # Use the configured path relative to project root
        db_path = os.path.join(ROOT, "dds_rag.db")

    seed(db_path, args.clear)


if __name__ == "__main__":
    main()
