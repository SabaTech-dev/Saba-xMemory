# xMemory Architecture

## Overview

xMemory implements a complete memory lifecycle for AI agents, built on PostgreSQL with pgvector for semantic search.

## Pipeline: Retain → Recall → Consolidate → Evolve

### 1. Retain Pipeline

**Purpose:** Store new memories with fact extraction and confidence scoring

**Steps:**
1. **LLM Extract**: Extract structured facts from input text
2. **Fact Filter**: Remove low-confidence facts (< 0.5)
3. **Confidence Tagger**: Assign confidence score based on source reliability
4. **Embedding**: Compute vector embedding (OpenAI/Ollama)
5. **Store**: Insert into `memory_units` table

**Data Model:**
```sql
memory_units (
  id UUID PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(1536),
  confidence DECIMAL(3,2),
  bank_id VARCHAR(50),
  created_at TIMESTAMP,
  archived_at TIMESTAMP NULL
)
```

### 2. Recall Pipeline

**Purpose:** Retrieve relevant memories with semantic search and graph expansion

**Steps:**
1. **Embed Query**: Convert query to vector
2. **Semantic Search**: Find top-k memories by cosine similarity
3. **Graph Expansion**: Follow knowledge links (entity, temporal, semantic)
4. **Rerank**: Re-score results with graph context
5. **Filter**: Apply confidence threshold (default: 0.7)

**Algorithm:**
```python
def recall(query, limit=10, min_confidence=0.7):
    # Step 1: Semantic search
    candidates = vector_search(query, limit=limit * 3)

    # Step 2: Graph expansion
    expanded = []
    for c in candidates:
        links = get_links(c['id'], types=['entity', 'semantic'])
        expanded.extend(links)

    # Step 3: Rerank with graph context
    results = rerank(candidates + expanded, query)

    # Step 4: Filter by confidence
    return [r for r in results if r['confidence'] >= min_confidence]
```

### 3. Consolidate Pipeline

**Purpose:** Maintain memory quality through deduplication and archiving

**Steps:**
1. **Cross-Bank Link**: Connect related memories across different agent banks
2. **Deduplicate**: Merge duplicate memories (11.45% dedup rate)
3. **Temporal Archive**: Archive stale memories (> 30 days unused)

**Deduplication Logic:**
```python
def is_duplicate(mem1, mem2):
    # Same embedding similarity
    if cosine_similarity(mem1['embedding'], mem2['embedding']) > 0.95:
        return True

    # Same content + similar confidence
    if (mem1['content'] == mem2['content'] and
        abs(mem1['confidence'] - mem2['confidence']) < 0.1):
        return True

    return False
```

### 4. Evolve Pipeline

**Purpose**: Periodic re-embedding and confidence decay

**Steps:**
1. **Re-embed**: Update embeddings for old memories
2. **Confidence Decay**: Reduce confidence over time
3. **Temporal Invalidation**: Invalidate outdated facts

**Decay Formula:**
```python
def decay_confidence(memory, days_ago):
    decay_factor = 0.95 ** (days_ago / 30)  # 5% decay per 30 days
    return memory['confidence'] * decay_factor
```

## Schema Reference

### Tables

#### memory_units
Stores individual memory units with embeddings and metadata.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| content | TEXT | Memory content |
| embedding | vector(1536) | Semantic embedding |
| confidence | DECIMAL(3,2) | Confidence score (0-1) |
| bank_id | VARCHAR(50) | Bank identifier (agent isolation) |
| created_at | TIMESTAMP | Creation time |
| archived_at | TIMESTAMP NULL | Archive time (if archived) |

#### memory_links
Stores knowledge graph links between memories.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| source_id | UUID | Source memory unit |
| target_id | UUID | Target memory unit |
| link_type | VARCHAR(20) | Link type (entity, temporal, semantic) |
| confidence | DECIMAL(3,2) | Link confidence |

#### banks
Stores memory bank metadata.

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(50) | Primary key (bank identifier) |
| name | VARCHAR(100) | Bank name |
| created_at | TIMESTAMP | Creation time |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_URL | postgresql://localhost/xmemory | Database connection URL |
| EMBEDDING_MODEL | text-embedding-3-small | Embedding model |
| EMBEDDING_PROVIDER | openai | Provider (openai, ollama) |
| MIN_CONFIDENCE | 0.7 | Minimum confidence for recall |
| DEDUP_THRESHOLD | 0.95 | Similarity threshold for dedup |
| ARCHIVE_DAYS | 30 | Days before archiving |

## Performance Considerations

### Vector Search
- Use pgvector's HNSW index for fast similarity search
- Typical latency: 10-50ms for 100k vectors

### Graph Expansion
- Depth-limited to 3 hops to prevent explosion
- Maximum candidates: 100 before reranking

### Deduplication
- Batch processing: compare new memories vs last 1000 stored
- 11.45% dedup rate reduces storage overhead

## See Also

- [README.md](../README.md) - Quick start and API reference
- [benchmarks/](../benchmarks/) - Reproducible benchmark scripts
