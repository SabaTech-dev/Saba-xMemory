# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Hybrid recall: vector + full-text search fused via Reciprocal Rank Fusion;
  explicit `search="vector"` without an embedding raises `ValueError`, auto mode
  degrades to full-text instead of silently returning empty results.
- Pluggable embedding providers (`xmemory/embeddings.py`): protocol +
  Ollama + OpenAI-compatible endpoints + deterministic local fallback;
  auto-embed on retain/recall.
- Real benchmark harness (`benchmarks/`): LoCoMo and EverMemBench runners
  (dataset download, ingest, QA, LLM judge), reproducible via scripts.
- Retrieval-time dedup: near-identical memories (cosine > 0.95 or equal
  normalized text) are collapsed after RRF fusion so top-k slots carry diverse
  evidence (`RECALL_DEDUP_THRESHOLD`).
- Confidence support in `retain(confidence=...)` and the evolve pipeline:
  `decay_confidence()` (5% per 30 days) and `MemoryBank.evolve()` applying it
  to scored memories.
- Benchmark harness: neighbor-context expansion of recalled turns, optional
  session-level chunking (`--chunk N`), ablation flag (`--no-neighbors`), and a
  date-aware LLM judge (relative date expressions resolved before judging).

### Changed
- Benchmark scoring pipeline hardened (semantic judge with date awareness).

### Fixed
- Vector recall path now works with plain lists (previously never executed
  with ndarray embeddings).
