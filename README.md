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
)

# Retain a memory
bank.retain("User prefers dark mode in all applications")

# Recall relevant memories
results = bank.recall("What are the user's UI preferences?")
for memory in results:
    print(memory.text)

# Get memory stats
stats = bank.stats()
print(f"Total memories: {stats.total_memories}, Avg confidence: {stats.avg_confidence}")
```

## Core Features

### 1. Semantic Recall

Use lexical search immediately, or semantic search when embeddings are available:

```python
bank.recall("Python 3.11")
bank.recall(
    "UI preferences",
    query_embedding=[0.1] * 384,
)
```

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

### 3. Persistence and Filtering

Memories can be stored with fact types, tags, and context:

```python
bank.retain(
    "Server is running version 2.1",
    fact_type="world",
    context="Deployment note",
    tags=["ops", "server"],
)
```

### 4. Graph Traversal

Navigate related memories through a knowledge graph:

```sql
SELECT * FROM graph_recall(
  query_embedding => (SELECT embedding FROM memory_units WHERE id = 'target-id'),
  query_bank_id => 'my-agent',
  top_k => 10,
  expansion_depth => 2,
  min_weight => 0.5
);
```

### 5. Cross-Session Consolidation

The schema supports cross-bank linking through `memory_links`, but any higher-level consolidation workflow should be verified against the current code before relying on it.

## API Reference

### Memory Operations

| Operation | Method | Description |
|-----------|--------|-------------|
| `recall` | POST | Lexical search by default, semantic search when embeddings are provided |
| `retain` | POST | Store new memories with fact type, context, tags, and optional embedding |
| `list` | GET | List memories with filtering and pagination |
| `stats` | GET | Memory statistics (count, confidence, links) |
| `delete` | DELETE | Delete a memory by ID within the active bank |

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
