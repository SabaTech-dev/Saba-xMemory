# xMemory Framework — Persistent Semantic Memory for AI Agents

![CI](https://github.com/SabaTech-dev/Saba-xMemory/workflows/CI/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/SabaTech-dev/Saba-xMemory)
![License](https://img.shields.io/badge/license-MIT-blue)

## Abstract

xMemory is a semantic memory framework for AI agents, enabling persistent, retrievable, and evolvable knowledge storage. Unlike simple vector stores, xMemory implements a memory lifecycle: retain, recall, and introspection (stats, listing, deletion). Built on PostgreSQL with pgvector, it has powered OpenClaw's multi-agent ecosystem in production.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  xMemory Framework                    │
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
└─────────────────────────────────────────────────────────┘
```

### Pipeline Overview

- **Retain**: store memories with confidence scores and embeddings (implemented in `MemoryBank.retain`)
- **Recall**: semantic search with embedding similarity (implemented in `MemoryBank.recall`)
- **Introspection**: listing, aggregate stats, deletion (`list_memories`, `stats`, `delete`)
- **Consolidate** (roadmap): cross-bank knowledge links, deduplication, stale-memory archive — not yet implemented in this library
- **Evolve** (roadmap): periodic re-embedding, confidence decay, temporal invalidation — not yet implemented in this library

## Benchmarks

> ⚠️ **Historical results — not reproducible from this repository.** The numbers
> below come from a previous internal evaluation (Hindsight/xMemory adapter,
> October 2026). The scripts in [benchmarks/](benchmarks/) are placeholders
> pending a public, self-contained benchmark harness. Do not treat them as
> current claims.

| Benchmark | Historical result | Target |
|-----------|-------------------|--------|
| LoCoMo | 62.1% | 69.7% |
| EverMemBench | 45.4% | 51.6% |

## Historical Production Stats

Deployed in the OpenClaw multi-agent system (private deployment, snapshot as of
October 2026). These figures describe that deployment, not this package:

| Metric | Value |
|--------|-------|
| Total memories | 11,712 |
| Active memories | 8,214 |
| Archived memories | 3,498 |
| Knowledge links | 820,016 |
| Cross-bank links | 34,168 |
| Avg confidence score | 0.953 |
| Deduplication rate | 11.45% |
| Archive rate | 29.86% |

## Quick Start

```bash
git clone https://github.com/SabaTech-dev/Saba-xMemory.git
cd Saba-xMemory

# Requires PostgreSQL with pgvector and pg_trgm extensions
# Apply the schema:
psql "$DATABASE_URL" -f sql/schema.sql
psql "$DATABASE_URL" -f sql/functions.sql

# Install
pip install -e .
```

```python
from xmemory import MemoryBank

bank = MemoryBank(bank_id="my-agent", db_url="postgresql://localhost/xmemory")
bank.ensure_bank(mission="Assistant memory")

result = bank.retain("The user prefers PostgreSQL for analytics workloads")
print(result.id, result.fact_type)

hits = bank.recall(
    query="which database does the user prefer?",
    top_k=5,
)
for mem in hits.memories:
    print(mem.text, mem.confidence)
```

## API Reference

`xmemory.MemoryBank` — Python API over PostgreSQL + pgvector:

| Method | Description |
|--------|-------------|
| `ensure_bank(mission)` | Create or update the memory bank |
| `retain(content, fact_type, context, tags, embedding)` | Store a memory. `embedding` optional when an embedding provider is configured (auto-generated) |
| `recall(query, top_k, query_embedding, fact_types, search)` | Hybrid ranking: vector similarity fused with full-text search (RRF). `search="auto"` (default) degrades to full-text when no embedding is available; `search="vector"` requires an embedding and raises `ValueError` otherwise; `search="text"` is full-text only |
| `list_memories(limit, offset, ...)` | Paginated listing with filters |
| `stats()` | Aggregate statistics (counts, confidence distribution) |
| `delete(memory_id)` | Delete a memory |

### Example: Recall

```python
from xmemory import MemoryBank

bank = MemoryBank(bank_id="my-agent", db_url="postgresql://localhost/xmemory")

result = bank.recall(
    query="What did we discuss about the database schema?",
    top_k=10,
    query_embedding=embed(query),  # provide your own embedding vector
)

for mem in result.memories:
    print(f"{mem.text[:50]}... (confidence: {mem.confidence})")
```

### Embedding Providers

Embeddings are pluggable and resolved local-first:

1. Explicit: `MemoryBank(..., embedding_provider=OllamaEmbeddingProvider())` (or any object with `embed`/`dim`).
2. `XMEMORY_EMBEDDING_PROVIDER` env var: `ollama` | `openai` | `hash` | `none` (disables auto-embedding).
3. Auto-detect: local Ollama server, then OpenAI (`OPENAI_API_KEY`), then the built-in `HashEmbeddingProvider`.

The hash provider works offline with zero setup (deterministic, lexical-only quality — fine for bootstrapping and tests; use Ollama or OpenAI for semantic quality). The default schema stores `vector(384)`; the OpenAI default model produces 1536 dims, so create the column with the matching dimension if you use it.

```python
from xmemory import MemoryBank, OllamaEmbeddingProvider

bank = MemoryBank(
    bank_id="my-agent",
    db_url="postgresql://localhost/xmemory",
    embedding_provider=OllamaEmbeddingProvider(model="nomic-embed-text"),
)
bank.retain("User prefers dark mode")          # embedding generated automatically
result = bank.recall("UI preferences")          # query embedded automatically
```

## Data Flow (implemented core)

```
Input Text ──────────────────────► memory_units (store, with optional embedding)
                                                 │
Query Embedding ──► Vector Similarity (pgvector) ─┘──► Ranked Results
```

Planned stages not yet in the library: LLM fact extraction, embedding providers,
rerank, graph expansion, `memory_links` navigation (schema exists in `sql/`).

## Roadmap

### v1.1 (Next)
- [ ] Improved LoCoMo score (target: 65%+)
- [ ] Streaming recall for large result sets
- [ ] Multi-modal memory (images, audio)

### v1.2 (Future)
- [ ] Federated memory across distributed agents
- [ ] Automatic schema evolution
- [ ] Memory export/import formats

## Documentation

- [Architecture](docs/architecture.md) - Detailed pipeline descriptions and schema reference
- [Contributing](CONTRIBUTING.md) - Development setup and guidelines

## License

MIT License — see [LICENSE](LICENSE)
