"""
Data models for agent-memory-store.
Uses only Python stdlib — no external dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """
    Classification of memory by its role in agent cognition.

    Modelled on cognitive science memory taxonomy:
    - EPISODIC:   Specific events ("user asked about X on Monday")
    - SEMANTIC:   General facts ("user works in healthcare")
    - PROCEDURAL: How to do things ("user prefers step-by-step explanations")
    - WORKING:    Temporary context for the current task (auto-expires)
    """
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


@dataclass
class Memory:
    """
    A single memory unit stored in the memory store.

    Attributes:
        id:           Unique memory identifier (auto-generated)
        content:      The memory content (plain text)
        memory_type:  Classification (episodic/semantic/procedural/working)
        agent_id:     Which agent created this memory
        user_id:      Which user this memory belongs to (for multi-user systems)
        tags:         List of tags for filtering
        importance:   Importance weight 0.0–1.0 (higher = retrieved more often)
        created_at:   When this memory was created
        accessed_at:  When this memory was last retrieved
        expires_at:   When this memory expires (None = never)
        access_count: How many times this memory has been retrieved
        metadata:     Additional structured data (stored as JSON)
    """

    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    agent_id: str = "default"
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime | None = None
    expires_at: datetime | None = None
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """True if this memory has passed its expiry time."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def __repr__(self) -> str:
        tags_str = f" tags={self.tags}" if self.tags else ""
        return (
            f"Memory(id={self.id!r}, type={self.memory_type.value}, "
            f"importance={self.importance:.2f}{tags_str}, "
            f"content={self.content[:60]!r}{'...' if len(self.content) > 60 else ''})"
        )


@dataclass
class SearchResult:
    """A memory returned from a search query, with relevance score."""

    memory: Memory
    relevance_score: float  # 0.0–1.0, higher = more relevant

    # Delegate common attribute access to the wrapped memory
    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def content(self) -> str:
        return self.memory.content

    @property
    def memory_type(self) -> MemoryType:
        return self.memory.memory_type

    @property
    def tags(self) -> list[str]:
        return self.memory.tags

    @property
    def importance(self) -> float:
        return self.memory.importance

    @property
    def agent_id(self) -> str:
        return self.memory.agent_id

    def __repr__(self) -> str:
        return f"SearchResult(score={self.relevance_score:.3f}, {self.memory!r})"


class AuditAction(str, Enum):
    """Actions recorded in the audit log."""
    ADD = "add"
    SEARCH = "search"
    FORGET = "forget"
    UPDATE = "update"
    EXPIRE = "expire"
    SUMMARISE = "summarise"


@dataclass
class AuditEvent:
    """
    A single entry in the audit log.

    Every operation on the memory store creates an audit event,
    providing a complete, tamper-evident history of all memory operations.
    """

    action: AuditAction
    agent_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    memory_id: str | None = None
    details: str = ""
    user_id: str | None = None
    id: str = field(default_factory=lambda: f"audit_{uuid.uuid4().hex[:12]}")

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        target = f" memory={self.memory_id}" if self.memory_id else ""
        return f"AuditEvent({ts} | {self.action.value} | agent={self.agent_id}{target} | {self.details})"
