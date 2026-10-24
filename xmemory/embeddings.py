"""
Embedding providers for xMemory.

A provider turns text into a fixed-length float vector. Providers are
pluggable and resolved with a local-first default chain:

1. Explicit argument (``get_provider("ollama")`` or passing an instance
   to ``MemoryBank(embedding_provider=...)``).
2. ``XMEMORY_EMBEDDING_PROVIDER`` environment variable (``ollama``,
   ``openai`` or ``hash``).
3. Auto-detection: Ollama if a local server answers, then OpenAI if
   ``OPENAI_API_KEY`` is set, otherwise the built-in :class:`HashEmbeddingProvider`.

The hash provider works offline with zero setup. It is deterministic and
reproducible, but it captures lexical overlap only — for real semantic
quality configure Ollama or OpenAI.
"""

import hashlib
import math
import os
from typing import Optional, Protocol


class EmbeddingProvider(Protocol):
    """Anything that can turn text into a fixed-length vector."""

    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbeddingProvider:
    """Deterministic, dependency-free embedding based on character n-gram hashing.

    Quality: lexical. Use for zero-setup bootstrapping and tests; configure a
    real provider (Ollama/OpenAI) for semantic search quality.
    """

    def __init__(self, dim: int = 384, ngram: int = 3):
        self.name = "hash"
        self.dim = dim
        self.ngram = ngram

    def _validate_dim(self, expected: Optional[int]) -> None:
        if expected is not None and expected != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: provider '{self.name}' produces "
                f"{self.dim} dims but the target expects {expected}. Use a "
                f"provider with dim={expected} or change the schema column type."
            )

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = text.lower().strip()
        for i in range(max(1, len(text) - self.ngram + 1)):
            gram = text[i : i + self.ngram]
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OllamaEmbeddingProvider:
    """Embeddings from a local Ollama server (default model: nomic-embed-text)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
    ):
        self.name = "ollama"
        self.model = model
        self.base_url = (
            base_url or os.environ.get("XMEMORY_OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dim probe"))
        return self._dim

    def _validate_dim(self, expected: Optional[int]) -> None:
        if expected is not None and self.dim != expected:
            raise ValueError(
                f"embedding dimension mismatch: provider '{self.name}' model "
                f"'{self.model}' produces {self.dim} dims but the target expects "
                f"{expected}. Use a provider with dim={expected} or change the "
                f"schema column type."
            )

    def embed(self, text: str) -> list[float]:
        import requests

        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbeddingProvider:
    """Embeddings from the OpenAI API (default model: text-embedding-3-small, 1536 dims).

    Note: the default schema uses vector(384); with this provider create the
    column with the matching dimension (see sql/schema.sql).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        self.name = "openai"
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._dim = dim  # resolved lazily on first embed if not set

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dim probe"))
        return self._dim

    def _validate_dim(self, expected: Optional[int]) -> None:
        if expected is not None and self.dim != expected:
            raise ValueError(
                f"embedding dimension mismatch: provider '{self.name}' model "
                f"'{self.model}' produces {self.dim} dims but the target expects "
                f"{expected}. Use a provider with dim={expected} or change the "
                f"schema column type."
            )

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, text: str) -> list[float]:
        resp = self._client().embeddings.create(model=self.model, input=[text])
        vec = resp.data[0].embedding
        if self._dim is None:
            self._dim = len(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client().embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if self._dim is None and vecs:
            self._dim = len(vecs[0])
        return vecs


def _detect_ollama() -> bool:
    """True if a local Ollama server answers quickly. Never raises: any
    failure (missing ``requests``, connection refused, timeout) means False."""
    try:
        import requests
    except ImportError:
        return False
    try:
        requests.get(
            os.environ.get("XMEMORY_OLLAMA_URL", "http://localhost:11434") + "/api/tags",
            timeout=1.5,
        )
        return True
    except Exception:
        return False


def _has_openai_key() -> bool:
    """True only if a key AND the openai package are available, so the
    auto-detect chain degrades to the hash provider when the lib is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


_PROVIDERS = {
    "hash": HashEmbeddingProvider,
    "ollama": OllamaEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
}


def get_provider(preferred: Optional[str] = None) -> EmbeddingProvider:
    """Resolve an embedding provider.

    Order: explicit ``preferred`` > ``XMEMORY_EMBEDDING_PROVIDER`` env var >
    auto-detect (Ollama, then OpenAI key, then hash fallback).
    """
    choice = preferred or os.environ.get("XMEMORY_EMBEDDING_PROVIDER")
    if choice:
        if choice.lower() == "none":
            raise ValueError("embedding provider disabled via XMEMORY_EMBEDDING_PROVIDER=none")
        cls = _PROVIDERS.get(choice.lower())
        if cls is None:
            raise ValueError(
                f"unknown embedding provider '{choice}'. Available: {', '.join(sorted(_PROVIDERS))}"
            )
        return cls()

    if _detect_ollama():
        return OllamaEmbeddingProvider()
    if _has_openai_key():
        return OpenAIEmbeddingProvider()
    return HashEmbeddingProvider()
