"""
MemoryBank — Core interface for xMemory Framework.

Provides high-level operations: retain, recall, list, stats, consolidate.
"""

import os
import time
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector

from xmemory.models import Memory, MemoryStats, RecallResult, RetainRequest, RetainResult


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
    """

    def __init__(
        self,
        bank_id: str,
        db_url: Optional[str] = None,
        embedding_dim: int = 384,
    ):
        self.bank_id = bank_id
        self.db_url = db_url or os.environ.get("XMEMORY_DB_URL", "")
        self.embedding_dim = embedding_dim
        self._conn: Optional[psycopg.Connection] = None

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

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_units (bank_id, text, fact_type, context, tags, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (self.bank_id, content, fact_type, context, tags, embedding),
            )
            row = cur.fetchone()

        return RetainResult(
            id=str(row[0]),
            text=content,
            fact_type=fact_type,
        )

    # ─── Recall ───────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[list[float]] = None,
        fact_types: Optional[list[str]] = None,
    ) -> RecallResult:
        """
        Semantic search for relevant memories.

        Uses vector similarity (cosine) for ranking.
        Optionally expands results via graph traversal.
        """
        start = time.time()
        conn = self._get_conn()
        memories = []

        if query_embedding is not None:
            with conn.cursor() as cur:
                fact_filter = ""
                params: list = [self.bank_id, query_embedding, top_k]
                if fact_types:
                    placeholders = ",".join(["%s"] * len(fact_types))
                    fact_filter = f" AND fact_type IN ({placeholders})"
                    params = [self.bank_id, query_embedding] + fact_types + [top_k]

                cur.execute(
                    f"""
                    SELECT id, text, fact_type, confidence_score, context,
                           tags, mentioned_at, created_at, access_count, metadata
                    FROM memory_units
                    WHERE bank_id = %s
                      AND embedding IS NOT NULL
                      {fact_filter}
                    ORDER BY embedding <=> %s
                    LIMIT %s
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
