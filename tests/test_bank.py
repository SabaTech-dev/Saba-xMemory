"""
Tests for xMemory Framework — MemoryBank operations.

Requires a running PostgreSQL with pgvector and the schema applied.
Set XMEMORY_DB_URL or use the default test database.
"""

import os
import pytest
from xmemory import MemoryBank, Memory, MemoryStats
from xmemory.bank import MemoryBank as MB


# ─── Fixtures ──────────────────────────────────────────────────────────

BANK_ID = "test-bank"
DB_URL = os.environ.get("XMEMORY_DB_URL", "postgresql://localhost/xmemory_test")


@pytest.fixture(scope="module")
def bank():
    """Create a test memory bank."""
    b = MemoryBank(bank_id=BANK_ID, db_url=DB_URL)
    try:
        b.ensure_bank(mission="Test bank for xMemory unit tests")
    except Exception:
        pytest.skip("PostgreSQL not available — set XMEMORY_DB_URL")
    yield b
    # Cleanup
    try:
        conn = b._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_units WHERE bank_id = %s", (BANK_ID,))
    except Exception:
        pass
    b.close()


# ─── Retain Tests ─────────────────────────────────────────────────────

class TestRetain:
    def test_retain_basic(self, bank):
        result = bank.retain("Test memory: Python is great for AI")
        assert result.id is not None
        assert result.text == "Test memory: Python is great for AI"
        assert result.fact_type == "observation"

    def test_retain_with_fact_type(self, bank):
        result = bank.retain(
            "PostgreSQL supports vector search via pgvector",
            fact_type="world",
        )
        assert result.fact_type == "world"

    def test_retain_with_tags(self, bank):
        result = bank.retain(
            "Deployed v1.0 to production",
            tags=["deployment", "production"],
        )
        assert result.deduplicated is False

    def test_retain_with_context(self, bank):
        result = bank.retain(
            "User prefers dark mode",
            context="User settings conversation",
        )
        assert result.text == "User prefers dark mode"


# ─── Recall Tests ─────────────────────────────────────────────────────

class TestRecall:
    def test_recall_basic(self, bank):
        # First retain something
        bank.retain("xMemory supports semantic recall")
        # Then recall
        result = bank.recall("semantic recall")
        assert isinstance(result, object)
        assert result.query == "semantic recall"

    def test_recall_returns_memories(self, bank):
        result = bank.recall("test")
        assert isinstance(result.memories, list)

    def test_recall_has_latency(self, bank):
        result = bank.recall("test")
        assert result.latency_ms >= 0

    def test_recall_without_embedding_uses_lexical_fallback(self, monkeypatch):
        class FakeCursor:
            def __init__(self):
                self.executed = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, query, params):
                self.executed.append((query, params))

            def fetchall(self):
                return [
                    (
                        "123",
                        "User prefers dark mode",
                        "observation",
                        None,
                        "Settings",
                        ["ui"],
                        None,
                        None,
                        0,
                        {},
                    )
                ]

        class FakeConnection:
            def __init__(self):
                self.cursor_obj = FakeCursor()

            def cursor(self):
                return self.cursor_obj

        fake_conn = FakeConnection()
        bank = MB(bank_id="test-bank", db_url="postgresql://localhost/xmemory_test")
        monkeypatch.setattr(bank, "_get_conn", lambda: fake_conn)

        result = bank.recall("dark mode")

        assert result.total == 1
        assert result.memories[0].text == "User prefers dark mode"
        query, params = fake_conn.cursor_obj.executed[0]
        assert "ILIKE" in query
        assert params[1] == "%dark mode%"


# ─── List Tests ───────────────────────────────────────────────────────

class TestList:
    def test_list_memories(self, bank):
        bank.retain("List test memory 1")
        bank.retain("List test memory 2")
        memories = bank.list_memories(limit=10)
        assert isinstance(memories, list)
        assert len(memories) >= 2

    def test_list_with_fact_type_filter(self, bank):
        bank.retain("World fact", fact_type="world")
        memories = bank.list_memories(fact_type="world", limit=10)
        assert all(m.fact_type == "world" for m in memories)

    def test_list_with_limit(self, bank):
        memories = bank.list_memories(limit=2)
        assert len(memories) <= 2


# ─── Stats Tests ──────────────────────────────────────────────────────

class TestStats:
    def test_stats_basic(self, bank):
        stats = bank.stats()
        assert isinstance(stats, MemoryStats)
        assert stats.bank_id == BANK_ID
        assert stats.total_memories >= 0

    def test_stats_has_types(self, bank):
        stats = bank.stats()
        assert isinstance(stats.memories_by_type, dict)


# ─── Delete Tests ─────────────────────────────────────────────────────

class TestDelete:
    def test_delete_memory(self, bank):
        result = bank.retain("Memory to be deleted")
        deleted = bank.delete(result.id)
        assert deleted is True

    def test_delete_nonexistent(self, bank):
        deleted = bank.delete("00000000-0000-0000-0000-000000000000")
        assert deleted is False


# ─── Model Tests ──────────────────────────────────────────────────────

class TestModels:
    def test_memory_model(self):
        m = Memory(
            id="test-id",
            text="Test memory",
            fact_type="world",
            confidence=0.95,
        )
        assert m.text == "Test memory"
        assert m.confidence == 0.95

    def test_memory_stats_model(self):
        s = MemoryStats(
            bank_id="test",
            total_memories=100,
            active_memories=80,
        )
        assert s.total_memories == 100
