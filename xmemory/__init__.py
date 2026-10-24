"""
xMemory Framework — Persistent Semantic Memory for AI Agents.

A production-ready framework for giving AI agents long-term memory
with semantic recall, confidence scoring, deduplication, and graph traversal.
"""

__version__ = "1.0.0"
__author__ = "Jose Manuel Sabarís García"
__license__ = "MIT"

from xmemory.bank import MemoryBank
from xmemory.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_provider,
)
from xmemory.models import Memory, MemoryStats, RecallResult

__all__ = [
    "MemoryBank",
    "Memory",
    "MemoryStats",
    "RecallResult",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_provider",
]
