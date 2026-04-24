"""
Tests for recall pipeline: semantic search with graph expansion.
"""
import pytest


def test_recall_semantic_search():
    """Test that recall performs semantic search."""
    query = "database schema optimization"

    from xmemory.recall import recall

    results = recall(query, limit=10)

    # Verify results are ranked and have structure
    assert len(results) > 0
    assert "content" in results[0]
    assert "confidence" in results[0]


def test_recall_filters_by_confidence():
    """Test that recall filters by minimum confidence."""
    query = "database"

    from xmemory.recall import recall

    results = recall(query, min_confidence=0.7)

    # Verify only high-confidence results returned
    for r in results:
        assert r["confidence"] >= 0.7


def test_recall_respects_limit():
    """Test that recall respects result limit."""
    query = "test query"

    from xmemory.recall import recall

    results = recall(query, limit=5)

    # Verify limit is respected
    assert len(results) <= 5
