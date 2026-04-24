"""
Tests for retain pipeline: storing memories with fact extraction and confidence.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_retain_basic_store(sample_memory, mock_db_connection):
    """Test that a basic memory can be stored."""
    # Mock the database insertion
    mock_cursor = mock_db_connection.cursor.return_value
    mock_cursor.fetchone.return_value = (sample_memory["id"],)

    # Import and call retain
    from xmemory.retain import retain

    result = retain(sample_memory, connection=mock_db_connection)

    # Verify memory was stored
    mock_cursor.execute.assert_called_once()
    assert result["status"] == "stored"
    assert result["confidence"] == 0.85


def test_retain_filters_low_confidence(sample_memory_low_confidence, mock_db_connection):
    """Test that low-confidence memories are filtered out."""
    mock_cursor = mock_db_connection.cursor.return_value

    from xmemory.retain import retain

    # Attempt to retain low-confidence memory
    result = retain(sample_memory_low_confidence, connection=mock_db_connection)

    # Verify database was NOT called for low confidence
    assert not mock_cursor.execute.called
    assert result["status"] == "filtered"
    assert result["reason"] == "low_confidence"


def test_retain_assigns_confidence():
    """Test that retain assigns confidence scores."""
    memory = {
        "content": "User asked about PostgreSQL performance tuning.",
        "bank_id": "test_agent"
    }

    from xmemory.retain import retain

    # Retain without confidence - should assign default
    result = retain(memory)

    assert result["confidence"] == 0.85
