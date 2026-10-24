"""
Tests for the embedding provider layer (xmemory.embeddings).
"""

import os

import pytest


class TestHashProvider:
    def test_embed_deterministic_and_normalized(self):
        from xmemory.embeddings import HashEmbeddingProvider

        p = HashEmbeddingProvider()
        a = p.embed("user prefers dark mode")
        b = p.embed("user prefers dark mode")
        c = p.embed("completely different text about databases")
        assert a == b
        assert a != c
        assert len(a) == p.dim == 384
        norm = sum(x * x for x in a) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch(self):
        from xmemory.embeddings import HashEmbeddingProvider

        p = HashEmbeddingProvider()
        out = p.embed_batch(["one", "two"])
        assert len(out) == 2 and all(len(v) == p.dim for v in out)


class TestProviderResolution:
    def test_explicit_provider_wins(self):
        from xmemory.embeddings import get_provider, HashEmbeddingProvider

        assert isinstance(get_provider("hash"), HashEmbeddingProvider)

    def test_env_var_selects_provider(self, monkeypatch):
        from xmemory import embeddings

        monkeypatch.setenv("XMEMORY_EMBEDDING_PROVIDER", "hash")
        monkeypatch.setattr(embeddings, "_detect_ollama", lambda: True)
        monkeypatch.setattr(embeddings, "_has_openai_key", lambda: True)
        p = embeddings.get_provider()
        assert p.name == "hash"

    def test_unknown_provider_raises(self):
        from xmemory.embeddings import get_provider

        with pytest.raises(ValueError, match="provider"):
            get_provider("nonexistent-provider")

    def test_auto_detect_falls_back_to_hash(self, monkeypatch):
        from xmemory import embeddings

        monkeypatch.delenv("XMEMORY_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(embeddings, "_detect_ollama", lambda: False)
        p = embeddings.get_provider()
        assert p.name == "hash"


class TestBankIntegration:
    def test_recall_and_retain_auto_embed(self):
        """With a provider configured, recall/retain work without manual embeddings."""
        from xmemory.embeddings import HashEmbeddingProvider
        from xmemory import MemoryBank

        bank = MemoryBank(
            bank_id="provider-integration-test",
            db_url=os.environ.get("XMEMORY_DB_URL", "postgresql://localhost/xmemory_test"),
            embedding_provider=HashEmbeddingProvider(),
        )
        try:
            bank.ensure_bank()
            bank.retain("Auto-embed probe: the launch was on Tuesday")
            result = bank.recall("when was the launch?")
            assert isinstance(result.memories, list)
            assert any("Tuesday" in m.text for m in result.memories)
        finally:
            try:
                conn = bank._get_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM memory_units WHERE bank_id = %s",
                        ("provider-integration-test",),
                    )
            except Exception:
                pass
            bank.close()

    def test_dim_mismatch_raises(self):
        from xmemory.embeddings import HashEmbeddingProvider

        p = HashEmbeddingProvider(dim=768)
        with pytest.raises(ValueError, match="dimension"):
            p._validate_dim(384)
