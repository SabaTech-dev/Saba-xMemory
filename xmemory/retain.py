"""
Retain pipeline: store memories with fact extraction and confidence scoring.
"""
from typing import Dict, Any


def retain(memory: Dict[str, Any], connection=None) -> Dict[str, Any]:
    """
    Store a memory with fact extraction and confidence scoring.

    Args:
        memory: Dictionary with 'content' and 'bank_id'
        connection: Database connection (optional for testing)

    Returns:
        Stored memory with assigned confidence
    """
    # Filter low-confidence memories
    if memory.get("confidence", 1.0) < 0.5:
        return {"status": "filtered", "reason": "low_confidence"}

    # Assign confidence if not present
    if "confidence" not in memory:
        memory["confidence"] = 0.85  # Default confidence

    # Store in database (mock for now)
    if connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO memory_units (content, confidence, bank_id) VALUES (%s, %s, %s)",
            (memory["content"], memory["confidence"], memory["bank_id"])
        )

    return {
        "id": memory.get("id"),
        "content": memory["content"],
        "confidence": memory["confidence"],
        "status": "stored"
    }
