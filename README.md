# xMemory Framework — Persistent Semantic Memory for AI Agents

![CI](https://github.com/SabaTech-dev/Saba-xMemory/workflows/CI/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/SabaTech-dev/Saba-xMemory)
![License](https://img.shields.io/badge/license-MIT-blue)

## Abstract

xMemory is a production-grade semantic memory framework for AI agents, enabling persistent, retrievable, and evolvable knowledge storage. Unlike simple vector stores, xMemory implements a full memory lifecycle: retain, recall, consolidate, and evolve. Built on PostgreSQL with pgvector, it powers OpenClaw's multi-agent ecosystem with 11,712 memories across 820,016 knowledge links.

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

- **Retain**: Extract facts, filter noise, compute embeddings, store with confidence scores
- **Recall**: Semantic search with embedding similarity, rerank with graph context
- **Consolidate**: Build cross-bank knowledge links, deduplicate (11.45% rate), archive stale memories
- **Evolve**: Periodic re-embedding, confidence decay, temporal invalidation

## Benchmarks

| Benchmark | Result | Target | Status |
|-----------|--------|--------|--------|
| LoCoMo | 62.1% | 69.7% | ⚠️ -7.6% |
| EverMemBench | 45.4% | 51.6% | ⚠️ -6.2% |

*Benchmarks run on Hindsight/xMemory adapter. See [benchmarks/](benchmarks/) for reproducible scripts.*

## Production Stats

6 months continuous operation with OpenClaw multi-agent system:

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
# Clone repo
git clone https://github.com/SabaTech-dev/Saba-xMemory.git
cd Saba-xMemory

# Setup PostgreSQL + pgvector
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Run example
python examples/basic_usage.py
```

## API Reference

### Operations

| Operation | Method | Description |
|-----------|--------|-------------|
| `recall` | POST | Semantic search + graph expansion for relevant memories |
| `retain` | POST | Store memories with fact extraction and confidence scoring |
| `list` | GET | List memories with filters (by bank, confidence, time range) |
| `stats` | GET | Get aggregate statistics (counts, confidence distribution) |
| `graph` | GET | Retrieve knowledge graph data for visualization |

### Example: Recall

```python
from xmemory import XMemoryClient

client = XMemoryClient(db_url="postgresql://localhost/xmemory")

results = client.recall(
    query="What did we discuss about the database schema?",
    limit=10,
    min_confidence=0.7
)

for result in results:
    print(f"{result['content'][:50]}... (confidence: {result['confidence']:.2f})")
```

## Data Flow

```
Input Text → LLM Extraction → Fact Filter → Embedding
                                                           ↓
                                        memory_units (store)
                                                           ↓
Query Text → Embedding + Rerank → Graph Expansion → Ranked Results
                                                           ↓
                                       memory_links (navigation)
```

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
