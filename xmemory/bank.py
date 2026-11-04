"""
MemoryBank — Core interface for xMemory Framework.

Provides high-level operations: retain, recall, list, stats, consolidate.
"""

import os
import time
from typing import Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from xmemory.embeddings import EmbeddingProvider, HashEmbeddingProvider, get_provider
from xmemory.models import Memory, MemoryStats, RecallResult, RetainRequest, RetainResult

# Cosine similarity above which two retrieved memories are considered
# near-duplicates and collapsed into one (see docs/architecture.md, dedup logic).
RECALL_DEDUP_THRESHOLD = 0.95

# Confidence decay: 5% reduction per 30 days without a mention (see
# docs/architecture.md, evolve pipeline).
DECAY_RATE_PER_30D = 0.95


def decay_confidence(confidence: float, days_ago: float) -> float:
    """Decay a memory's confidence by age: 5% per 30 days since last mention."""
    return confidence * (DECAY_RATE_PER_30D ** (days_ago / 30.0))


class MemoryBank:
    """
    Persistent semantic memory bank backed by PostgreSQL + pgvector.

    Usage:
        bank = MemoryBank(
            bank_id="my-agent",
            db_url="postgresql://localhost/mydb",
        )
        bank.retain("User prefers dark mode")
        results = bank.recall("UI preferences")

    Embeddings: pass ``embedding_provider`` explicitly, or let the local-first
    default chain resolve one (see ``xmemory.embeddings``). When a provider is
    available, ``retain``/``recall`` generate embeddings automatically; without
    one, calls that need an embedding fail with an explicit ``ValueError``
    instead of returning empty results silently.
    """

    def __init__(
        self,
        bank_id: str,
        db_url: Optional[str] = None,
        embedding_dim: int = 384,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.bank_id = bank_id
        self.db_url = db_url or os.environ.get("XMEMORY_DB_URL", "")
        self.embedding_dim = embedding_dim
        self._provider: Optional[EmbeddingProvider] = embedding_provider
        self._conn: Optional[psycopg.Connection] = None

    @property
    def embedding_provider(self) -> Optional[EmbeddingProvider]:
        """Resolved embedding provider, lazily via the default chain."""
        if self._provider is None:
            try:
                self._provider = get_provider()
            except ValueError:
                return None
            self._provider._validate_dim(self.embedding_dim)
        return self._provider

    def _embed(self, text: str) -> Optional[list[float]]:
        """Embed text via the resolved provider; None if no provider is available."""
        provider = self.embedding_provider
        if provider is None:
            return None
        provider._validate_dim(self.embedding_dim)
        return provider.embed(text)

    # ─── Connection Management ────────────────────────────────────────

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.db_url, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ─── Bank Management ──────────────────────────────────────────────

    def ensure_bank(self, mission: str = "") -> None:
        """Create the bank if it doesn't exist."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO banks (bank_id, name, mission)
                VALUES (%s, %s, %s)
                ON CONFLICT (bank_id) DO UPDATE SET updated_at = NOW()
                """,
                (self.bank_id, self.bank_id, mission),
            )

    # ─── Retain ───────────────────────────────────────────────────────

    def retain(
        self,
        content: str,
        fact_type: str = "observation",
        context: str = "",
        tags: Optional[list[str]] = None,
        embedding: Optional[list[float]] = None,
        confidence: Optional[float] = None,
    ) -> RetainResult:
        """
        Store a new memory.

        In a full implementation, this would:
        1. Extract facts from content using LLM
        2. Generate embedding
        3. Check for semantic duplicates
        4. Assign confidence score
        5. Link to related memories

        For the framework skeleton, we store directly.
        """
        conn = self._get_conn()
        tags = tags or []
        if embedding is None:
            embedding = self._embed(content)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_units (bank_id, text, fact_type, context, tags, embedding, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    self.bank_id,
                    content,
                    fact_type,
                    context,
                    tags,
                    np.asarray(embedding, dtype=np.float32) if embedding is not None else None,
                    confidence,
                ),
            )
            row = cur.fetchone()

        return RetainResult(
            id=str(row[0]),
            text=content,
            fact_type=fact_type,
            confidence=confidence,
        )

    # ─── Recall ───────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[list[float]] = None,
        fact_types: Optional[list[str]] = None,
        search: str = "auto",
    ) -> RecallResult:
        """
        Semantic search for relevant memories.

        search modes:
            "auto"   — hybrid ranking (vector similarity fused with full-text
                       search via Reciprocal Rank Fusion). The vector leg is
                       skipped when no embedding is available (no argument and
                       no resolvable provider), falling back to full-text only.
            "vector" — vector similarity only; requires an embedding
                       (raises ValueError when none is available).
            "text"   — full-text search only.

        Optionally filters by fact_types.
        """
        start = time.time()
        conn = self._get_conn()
        memories = []

        if search not in ("auto", "vector", "text"):
            search = "auto"

        want_vector = search in ("auto", "vector")
        want_text = search in ("auto", "text")

        if want_vector:
            if query_embedding is None:
                query_embedding = self._embed(query)
            if query_embedding is None:
                if search == "vector":
                    raise ValueError(
                        "recall(search='vector') requires an embedding: pass "
                        "query_embedding=[...] or configure an embedding provider."
                    )
                want_vector = False  # auto mode: degrade to full-text

        candidate_k = max(top_k * 3, 30)
        fact_filter = ""
        fact_params: list = []
        if fact_types:
            placeholders = ",".join(["%s"] * len(fact_types))
            fact_filter = f" AND fact_type IN ({placeholders})"
            fact_params = list(fact_types)

        ranked: dict[str, tuple[float, Memory]] = {}
        embeddings: dict[str, Optional[np.ndarray]] = {}

        def add_leg(rows: list, weight: float = 1.0) -> None:
            for rank, row in enumerate(rows):
                mem = Memory(
                    id=str(row[0]),
                    text=row[1],
                    fact_type=row[2],
                    confidence=float(row[3]) if row[3] else None,
                    context=row[4] or "",
                    tags=row[5] or [],
                    mentioned_at=row[6],
                    created_at=row[7],
                    access_count=row[8] or 0,
                    metadata=row[9] or {},
                )
                if len(row) > 10 and row[10] is not None:
                    emb_raw = row[10].to_list() if hasattr(row[10], "to_list") else row[10]
                    embeddings.setdefault(mem.id, np.asarray(emb_raw, dtype=np.float32))
                rrf = 1.0 / (60.0 + rank + 1)
                key = mem.id
                score, prev = ranked.get(key, (0.0, mem))
                ranked[key] = (score + weight * rrf, prev or mem)

        select_cols = """
            id, text, fact_type, confidence_score, context,
            tags, mentioned_at, created_at, access_count, metadata, embedding
        """

        with conn.cursor() as cur:
            if want_vector and query_embedding is not None:
                query_vec = np.asarray(query_embedding, dtype=np.float32)
                cur.execute(
                    f"""
                    SELECT {select_cols}
                    FROM memory_units
                    WHERE bank_id = %s
                      AND embedding IS NOT NULL
                      {fact_filter}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    [self.bank_id, query_vec] + fact_params + [candidate_k],
                )
                add_leg(cur.fetchall())

            if want_text:
                # OR-of-words full-text over text + context (speaker/date live
                # in context); ts_rank_cd favors turns matching more terms.
                or_query = (
                    " OR ".join(w for w in query.replace("'", " ").split() if len(w) > 2) or query
                )
                cur.execute(
                    f"""
                    SELECT {select_cols}
                    FROM memory_units
                    WHERE bank_id = %s
                      AND to_tsvector('english', coalesce(text, '') || ' ' || coalesce(context, ''))
                          @@ websearch_to_tsquery('english', %s)
                      {fact_filter}
                    ORDER BY ts_rank_cd(
                        to_tsvector('english', coalesce(text, '') || ' ' || coalesce(context, '')),
                        websearch_to_tsquery('english', %s)) DESC
                    LIMIT %s
                    """,
                    [self.bank_id, or_query, or_query] + fact_params + [candidate_k],
                )
                add_leg(cur.fetchall())

        fused = sorted(ranked.items(), key=lambda kv: kv[1][0], reverse=True)

        # Retrieval-time dedup: collapse near-identical memories so top-k slots
        # carry diverse evidence instead of repeated variants of one fact.
        def _is_duplicate(a_id: str, a_text: str, kept: list[tuple[str, str]]) -> bool:
            a_emb = embeddings.get(a_id)
            a_norm = " ".join(a_text.lower().split())
            for b_id, b_text in kept:
                if a_norm == " ".join(b_text.lower().split()):
                    return True
                b_emb = embeddings.get(b_id)
                if a_emb is not None and b_emb is not None:
                    denom = float(np.linalg.norm(a_emb) * np.linalg.norm(b_emb))
                    if denom and float(np.dot(a_emb, b_emb)) / denom > RECALL_DEDUP_THRESHOLD:
                        return True
            return False

        kept_refs: list[tuple[str, str]] = []
        memories = []
        for _, (_, mem) in fused:
            if _is_duplicate(mem.id, mem.text, kept_refs):
                continue
            kept_refs.append((mem.id, mem.text))
            memories.append(mem)
            if len(memories) >= top_k:
                break

        elapsed = (time.time() - start) * 1000
        return RecallResult(
            query=query,
            memories=memories,
            total=len(memories),
            latency_ms=round(elapsed, 1),
        )

    # ─── List ─────────────────────────────────────────────────────────

    def list_memories(
        self,
        limit: int = 50,
        offset: int = 0,
        fact_type: Optional[str] = None,
        order_by: str = "created_at",
        order: str = "desc",
    ) -> list[Memory]:
        """List memories with filtering and pagination."""
        conn = self._get_conn()
        memories = []

        allowed_orders = {"created_at", "confidence_score", "mentioned_at"}
        if order_by not in allowed_orders:
            order_by = "created_at"
        direction = "DESC" if order == "desc" else "ASC"

        with conn.cursor() as cur:
            params: list = [self.bank_id, limit, offset]
            type_filter = ""
            if fact_type:
                type_filter = " AND fact_type = %s"
                params = [self.bank_id, fact_type, limit, offset]

            cur.execute(
                f"""
                SELECT id, text, fact_type, confidence_score, context,
                       tags, mentioned_at, created_at, access_count, metadata
                FROM memory_units
                WHERE bank_id = %s
                  {type_filter}
                ORDER BY {order_by} {direction}
                LIMIT %s OFFSET %s
                """,
                params,
            )
            for row in cur.fetchall():
                memories.append(
                    Memory(
                        id=str(row[0]),
                        text=row[1],
                        fact_type=row[2],
                        confidence=float(row[3]) if row[3] else None,
                        context=row[4] or "",
                        tags=row[5] or [],
                        mentioned_at=row[6],
                        created_at=row[7],
                        access_count=row[8] or 0,
                        metadata=row[9] or {},
                    )
                )

        return memories

    # ─── Stats ────────────────────────────────────────────────────────

    def stats(self) -> MemoryStats:
        """Get memory bank statistics."""
        conn = self._get_conn()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE confidence_score IS NULL OR confidence_score >= %s) as active,
                    COUNT(*) FILTER (WHERE confidence_score < %s) as archived,
                    AVG(confidence_score) as avg_conf
                FROM memory_units
                WHERE bank_id = %s
                """,
                (0.3, 0.3, self.bank_id),
            )
            row = cur.fetchone()

            cur.execute(
                """
                SELECT fact_type, COUNT(*)
                FROM memory_units WHERE bank_id = %s
                GROUP BY fact_type
                """,
                (self.bank_id,),
            )
            by_type = {r[0]: r[1] for r in cur.fetchall()}

            cur.execute(
                """
                SELECT ml.link_type, COUNT(*)
                FROM memory_links ml
                JOIN memory_units mu ON mu.id = ml.source_id
                WHERE mu.bank_id = %s
                GROUP BY ml.link_type
                """,
                (self.bank_id,),
            )
            links_by_type = {r[0]: r[1] for r in cur.fetchall()}

            total_links = sum(links_by_type.values())

        return MemoryStats(
            bank_id=self.bank_id,
            total_memories=row[0] or 0,
            active_memories=row[1] or 0,
            archived_memories=row[2] or 0,
            total_links=total_links,
            avg_confidence=float(row[3]) if row[3] else None,
            memories_by_type=by_type,
            links_by_type=links_by_type,
        )

    # ─── Delete ───────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory_units WHERE id = %s AND bank_id = %s",
                (memory_id, self.bank_id),
            )
            return cur.rowcount > 0

    # ─── Evolve ────────────────────────────────────────────────────────

    def evolve(self) -> int:
        """Apply confidence decay to scored memories of this bank.

        Each memory with a confidence score loses 5% per 30 days since its
        last mention (or creation when never mentioned). Memories without a
        confidence score are left untouched. Returns the number of updated
        memories.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_units
                SET confidence_score = confidence_score * power(
                    %s,
                    EXTRACT(EPOCH FROM (NOW() - COALESCE(mentioned_at, created_at)))
                        / 86400.0 / 30.0
                )
                WHERE bank_id = %s
                  AND confidence_score IS NOT NULL
                """,
                (DECAY_RATE_PER_30D, self.bank_id),
            )
            return cur.rowcount
