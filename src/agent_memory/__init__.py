"""
agent-memory-store
==================
Lightweight persistent memory for AI agents.

Zero mandatory dependencies. Drop into any agent in 3 lines.
SQLite backend, privacy controls, TTL expiry, full audit log.

Quick start:
    from agent_memory import MemoryStore

    mem = MemoryStore("agent.db")

    # Store a memory
    mem.add("user prefers concise responses", tags=["preference"], agent_id="assistant")

    # Recall relevant memories
    results = mem.search("response style", limit=3)
    for r in results:
        print(r.content, r.relevance_score)

    # Summarise memories for injection into a prompt
    context = mem.context_for("what does the user prefer?")
    print(context)  # ready to inject into system prompt

    # Full privacy lifecycle
    mem.forget(agent_id="assistant")         # forget everything for this agent
    mem.forget(older_than_days=30)           # forget anything older than 30 days
    mem.forget(memory_id="mem_abc123")       # forget one specific memory

    # Audit trail
    for event in mem.audit_log(limit=20):
        print(event)
"""

from agent_memory.store import MemoryStore
from agent_memory.models import Memory, MemoryType, AuditEvent, SearchResult

__version__ = "1.0.0"
__author__ = "Linda Oraegbunam"
__all__ = ["MemoryStore", "Memory", "MemoryType", "AuditEvent", "SearchResult"]
