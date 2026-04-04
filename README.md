# agent-memory-store

**Lightweight persistent memory for AI agents. Drop in, zero dependencies, privacy-first.**

SQLite backend. Four memory types. TTL expiry. Full audit log. Built-in `forget()` for GDPR compliance. Override `_score_relevance()` to plug in vector search.

[![PyPI](https://img.shields.io/badge/PyPI-agent--memory--store-blue?style=flat-square)](https://pypi.org/project/agent-memory-store/)
[![Tests](https://img.shields.io/badge/Tests-64%20passing-brightgreen?style=flat-square)](tests/)
[![Dependencies](https://img.shields.io/badge/Dependencies-zero%20mandatory-brightgreen?style=flat-square)](pyproject.toml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/-Linda_Oraegbunam-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/linda-oraegbunam/)

---

## Why not MemOS / MemU / A-MEM?

Every existing agent memory framework has the same problems: **complex setup** (Docker, Postgres, pgvector, embedding models), **no privacy controls**, **no audit log**, and they want to **own your entire agent architecture**.

agent-memory-store is different:

| | MemOS / MemU | agent-memory-store |
|---|---|---|
| Setup | Docker + Postgres + pgvector | `pip install agent-memory-store` |
| Dependencies | 20+ | **0 mandatory** |
| Backend | Postgres with pgvector | **SQLite (stdlib)** |
| `forget(user_id=...)` | ✗ | ✓ |
| Audit log | ✗ | ✓ |
| Drop into existing agent | Requires rewrite | **3 lines** |
| Vector search | Built-in | **Override one method** |

---

## Install

```bash
pip install agent-memory-store   # zero mandatory dependencies
```

---

## Quick Start

```python
from agent_memory import MemoryStore

mem = MemoryStore("agent.db")

# Store memories
mem.add("user prefers concise responses", tags=["preference"], importance=0.8)
mem.add("user works in NHS radiology", memory_type=MemoryType.SEMANTIC, importance=0.9)
mem.add("last session: discussed Python async", memory_type=MemoryType.EPISODIC)

# Get context for a prompt — ready to inject
context = mem.context_for("communication style")
system_prompt = f"You are a helpful assistant.\n\nWhat I know:\n{context}"

# Search
results = mem.search("response format", limit=3)
for r in results:
    print(f"{r.relevance_score:.2f}  {r.content}")

# Privacy: forget everything for a user
mem.forget(user_id="user_123")          # right to erasure

# Data minimisation
mem.forget(older_than_days=90)          # forget anything older than 90 days
mem.forget(tags=["sensitive"])          # forget by category

# Audit trail
for event in mem.audit_log():
    print(event)

# Stats
print(mem.stats())
```

---

## Memory Types

Modelled on cognitive science memory taxonomy:

```python
from agent_memory import MemoryType

MemoryType.SEMANTIC    # General facts: "user works in healthcare"
MemoryType.EPISODIC    # Specific events: "user asked about X on Monday"
MemoryType.PROCEDURAL  # How to do things: "user prefers step-by-step responses"
MemoryType.WORKING     # Temporary context for the current task (set ttl_days=1)
```

---

## Privacy Controls

Built for GDPR Article 17 (Right to Erasure) and UK GDPR:

```python
# Forget a specific user — right to erasure
deleted = mem.forget(user_id="user_123")
print(f"Deleted {deleted} memories")

# Forget everything for an agent
mem.forget(agent_id="my_assistant")

# Data minimisation — delete old memories
mem.forget(older_than_days=30)

# Category-based deletion
mem.forget(tags=["sensitive", "pii"])

# Auto-expire working memory
mem.add("temporary task context", memory_type=MemoryType.WORKING, ttl_days=1)
mem.expire_stale()  # call at session start
```

Every `forget()` call is written to the audit log.

---

## Audit Log

Every operation creates an immutable audit event:

```python
for event in mem.audit_log(limit=20):
    print(event)
# AuditEvent(2026-03-28 14:30:01 | add    | agent=assistant mem=mem_abc123 | type=semantic tags=['preference'])
# AuditEvent(2026-03-28 14:30:05 | search | agent=assistant | query='response style' hits=2)
# AuditEvent(2026-03-28 14:35:12 | forget | agent=assistant | user_id=user_123 deleted=4)

# Filter by agent or action
events = mem.audit_log(agent_id="assistant", action=AuditAction.FORGET)
```

---

## Plug in Vector Search

The default scorer uses keyword overlap. Override one method for semantic search:

```python
from agent_memory import MemoryStore, Memory

class SemanticMemoryStore(MemoryStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _score_relevance(self, query: str, memory: Memory) -> float:
        import numpy as np
        q = self.model.encode(query)
        m = self.model.encode(memory.content)
        cos_sim = float(np.dot(q, m) / (np.linalg.norm(q) * np.linalg.norm(m)))
        return cos_sim * 0.7 + memory.importance * 0.3

mem = SemanticMemoryStore("agent.db")
# Everything else works exactly the same
```

---

## Multi-Agent & Multi-User

```python
# Same database, different agents
mem = MemoryStore("shared.db")

mem.add("prefers formal tone", agent_id="sales_agent", user_id="user_001")
mem.add("technical background", agent_id="support_agent", user_id="user_001")

# Search is scoped by agent
sales_context = mem.context_for("tone", agent_id="sales_agent")
support_context = mem.context_for("background", agent_id="support_agent")
```

---

## CLI

```bash
agent-memory stats agent.db
agent-memory list agent.db --agent my_assistant
agent-memory search agent.db "user preferences"
agent-memory audit agent.db --limit 50
agent-memory forget agent.db --older-than 90
agent-memory export agent.db backup.json
```

---

## Related Projects

- **[llm-extract](https://github.com/obielin/llm-extract)** — Extract structured data from any document using LLMs
- **[agentic-alignment-toolkit](https://github.com/obielin/agentic-alignment-toolkit)** — Evaluate goal-alignment and oversight gaps in autonomous AI pipelines

---

**Linda Oraegbunam** | [LinkedIn](https://www.linkedin.com/in/linda-oraegbunam/) | [Twitter](https://twitter.com/Obie_Linda)
