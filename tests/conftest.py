"""
Pytest fixtures for xMemory tests.
"""
import pytest
from typing import Dict, Any
from unittest.mock import MagicMock


@pytest.fixture
def sample_memory() -> Dict[str, Any]:
    """Sample memory unit for testing."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": "The user asked about database schema optimization for PostgreSQL.",
        "embedding": [0.1] * 1536,  # Mock embedding
        "confidence": 0.85,
        "bank_id": "test_agent",
        "created_at": "2024-01-01T00:00:00Z",
        "archived_at": None
    }


@pytest.fixture
def sample_memory_low_confidence() -> Dict[str, Any]:
    """Sample memory with low confidence."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "content": "The user asked about database schema optimization for PostgreSQL.",
        "embedding": [0.1] * 1536,
        "confidence": 0.45,
        "bank_id": "test_agent",
        "created_at": "2024-01-01T00:00:00Z",
        "archived_at": None
    }


@pytest.fixture
def sample_duplicate_memory() -> Dict[str, Any]:
    """Sample memory that is duplicate of sample_memory."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440002",
        "content": "The user asked about database schema optimization for PostgreSQL.",
        "embedding": [0.1] * 1536,
        "confidence": 0.87,
        "bank_id": "test_agent",
        "created_at": "2024-01-01T01:00:00Z",
        "archived_at": None
    }


@pytest.fixture
def sample_link() -> Dict[str, Any]:
    """Sample knowledge graph link."""
    return {
        "id": "660e8400-e29b-41d4-a716-446655440000",
        "source_id": "550e8400-e29b-41d4-a716-446655440000",
        "target_id": "550e8400-e29b-41d4-a716-446655440001",
        "link_type": "entity",
        "confidence": 0.9
    }


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    return mock_conn
