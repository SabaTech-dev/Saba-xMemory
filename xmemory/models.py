"""
Data models for xMemory Framework.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Memory(BaseModel):
    """A single memory unit."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    text: str
    fact_type: str = "observation"
    confidence: Optional[float] = None
    tags: list[str] = Field(default_factory=list)
    context: str = ""
    mentioned_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)
    access_count: int = 0


class RecallResult(BaseModel):
    """Result from a recall query."""

    query: str
    memories: list[Memory]
    total: int
    latency_ms: float = 0.0


class MemoryStats(BaseModel):
    """Statistics for a memory bank."""

    bank_id: str
    total_memories: int = 0
    active_memories: int = 0
    archived_memories: int = 0
    total_links: int = 0
    avg_confidence: Optional[float] = None
    memories_by_type: dict[str, int] = Field(default_factory=dict)
    links_by_type: dict[str, int] = Field(default_factory=dict)


class RetainRequest(BaseModel):
    """Request to retain a new memory."""

    content: str
    fact_type: str = "observation"
    context: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class RetainResult(BaseModel):
    """Result from a retain operation."""

    id: str
    text: str
    fact_type: str
    confidence: Optional[float] = None
    deduplicated: bool = False
    archived_ids: list[str] = Field(default_factory=list)
