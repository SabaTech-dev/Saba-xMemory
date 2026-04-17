xMemory-framework
=================

Persistent semantic memory framework for AI agents — built on PostgreSQL + pgvector.

> **Status:** Production-ready · Battle-tested with OpenClaw agents

## What is xMemory?

xMemory is a framework for giving AI agents **long-term semantic memory**. It provides:

- **Semantic Recall** — Find relevant memories using natural language queries with vector similarity
- **Smart Retention** — Automatically extract facts from conversations (world knowledge, experiences, observations)
- **Graph Traversal** — Navigate memory relationships with expandable graph queries
- **Confidence Scoring** — Every memory has a confidence score that evolves over time
- **Temporal Invalidation** — Automatically archive outdated memories
- **Cross-Session Consolidation** — Link memories across different agents and channels
- **Semantic Deduplication** — Eliminate redundant memories automatically

## Architecture

```
┌─────────────────────────────────────────────────┐
│                xMemory Framework                  │
├──────────────┬──────────────┬───────────────────┤
│   Retain      │    Recall    │   Consolidate     │
│  Pipeline     │   Pipeline   │    Pipeline        │
├──────────────┼──────────────┼───────────────────┤
│  LLM Extract  │  Embedding   │  Cross-Bank        │
│  Fact Filter  │  + Rerank    │  Link Builder       │
│  Confidence   │  Graph       │  Deduplication      │
│  Tagger       │  Expansion   │  Temporal Archive   │
├──────────────┴──────────────┴───────────────────┤
│              PostgreSQL + pgvector                │
│         (embeddings, links, metadata)             │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- PostgreSQL 14+ with pgvector extension
- An LLM API (OpenAI-compatible, Anthropic, or local via Ollama)
- Embedding model (local or API)

### Installation

```bash
# Clone the repo
git clone https://github.com/llllJokerllll/xMemory-framework.git
cd xMemory-framework

# Set up the database
psql -f sql/schema.sql your_database

# Configure
cp .env.example .env
# Edit .env with your API keys and database URL

# Run the test suite
pip install -r requirements.txt
pytest tests/ -v
```

### Basic Usage

```python
from xmemory import MemoryBank

# Initialize a memory bank
bank = MemoryBank(
    bank_id="my-agent",
    db_url="postgresql://localhost/mydb",
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embed_model="text-embedding-3-small"
)

# Retain a memory
bank.retain("User prefers dark mode in all applications")

# Recall relevant memories
results = bank.recall("What are the user's UI preferences?")
for memory in results:
    print(f"[{memory.confidence:.2f}] {memory.text}")

# Get memory stats
stats = bank.stats()
print(f"Total memories: {stats.total}, Avg confidence: {stats.avg_confidence}")
```

## Core Features

### 1. Semantic Deduplication

Automatically detects and removes semantically similar memories:

```python
# These will be deduplicated
bank.retain("The project uses Python 3.11")
bank.retain("Python 3.11 is used for the project")  # → duplicate detected, skipped
```

**Result:** 11.45% memory reduction in production.

### 2. Confidence Scoring

Every memory gets a confidence score (0.0–1.0) based on:
- Recency and access frequency
- Number of corroborating sources
- Temporal stability

```python
results = bank.recall("deployment process")
high_confidence = [m for m in results if m.confidence >= 0.8]
```

**Production stats:** Average confidence 0.953 across 11,687 memories.

### 3. Temporal Invalidation

Memories are automatically archived when they become outdated:

```python
bank.retain("Server is running version 2.1")  # Later...
bank.retain("Server upgraded to version 2.3")  # → old memory archived
```

**Production stats:** 29.86% of memories naturally archived.

### 4. Graph Traversal

Navigate related memories through a knowledge graph:

```sql
SELECT * FROM graph_recall(
  embedding => (SELECT embedding FROM memory_units WHERE id = 'target-id'),
  bank_id => 'my-agent',
  top_k => 10,
  expansion_depth => 2,
  min_weight => 0.5
);
```

**Production stats:** 820,016 links (entity: 648K, temporal: 87K, semantic: 83K).

### 5. Cross-Session Consolidation

Link memories across different agents and channels:

```python
# Agent A retains
bank_a.retain("Database migration scheduled for Friday")

# Agent B discovers the link
results = bank_b.recall("any scheduled changes?")
# → Finds the migration memory via cross-bank link
```

**Production stats:** 34,168 cross-bank links across 24 memory banks.

## API Reference

### Memory Operations

| Operation | Method | Description |
|-----------|--------|-------------|
| `recall` | POST | Semantic search with optional graph expansion |
| `retain` | POST | Store new memories with automatic fact extraction |
| `list` | GET | List memories with filtering and pagination |
| `consolidate` | POST | Trigger cross-bank consolidation |
| `stats` | GET | Memory statistics (count, confidence, links) |
| `graph` | GET | Knowledge graph visualization data |
| `entities` | GET | Named entities extracted from memories |
| `tags` | GET | Memory tags and categories |

### Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| `world` | Factual knowledge | "PostgreSQL supports pgvector since v0.5.0" |
| `experience` | Event-based memories | "Deployed v2.1 on April 16, had config issues" |
| `observation` | Agent observations | "User tends to work late at night" |

## Production Stats

Collected from **6 months of continuous operation** with OpenClaw agents:

| Metric | Value |
|--------|-------|
| Total memories | 11,712 |
| Active memories | 8,214 |
| Archived memories | 3,498 |
| Knowledge links | 820,016 |
| Cross-bank links | 34,168 |
| Memory banks | 24 |
| Avg confidence | 0.953 |
| Deduplication rate | 11.45% |
| Archive rate | 29.86% |

## Benchmarks

```bash
# Run the benchmark suite
python benchmarks/run_benchmarks.py

# Results example:
# Recall latency (p50): 45ms
# Recall latency (p95): 120ms
# Recall latency (p99): 280ms
# Retain latency: 150ms (includes LLM extraction)
# Deduplication accuracy: 94.2%
# Graph expansion depth=2: +35% recall relevance
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XMEMORY_DB_URL` | PostgreSQL connection string | Required |
| `XMEMORY_LLM_PROVIDER` | LLM provider (openai, anthropic, ollama) | `openai` |
| `XMEMORY_LLM_MODEL` | Model for fact extraction | `gpt-4o-mini` |
| `XMEMORY_EMBED_MODEL` | Embedding model | `text-embedding-3-small` |
| `XMEMORY_EMBED_DIMENSIONS` | Embedding dimensions | `384` |
| `XMEMORY_RERANKER` | Reranker model (optional) | None |
| `XMEMORY_CONFIDENCE_THRESHOLD` | Min confidence to keep | `0.3` |
| `XMEMORY_DEDUP_THRESHOLD` | Similarity threshold for dedup | `0.92` |
| `XMEMORY_ARCHIVE_DAYS` | Days before temporal check | `30` |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Inspired by [Hindsight](https://github.com/vectorize-io/hindsight) memory system
- Battle-tested with [OpenClaw](https://github.com/openclaw/openclaw) agent framework
