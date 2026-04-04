"""
MemoryStore — the main interface for agent-memory-store.

Storage: SQLite (stdlib sqlite3, zero external dependencies).
Search: Keyword-based relevance scoring with importance weighting.
        Swap for vector search by subclassing and overriding _score_relevance().
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Generator

from agent_memory.models import (
    AuditAction,
    AuditEvent,
    Memory,
    MemoryType,
    SearchResult,
)


class MemoryStore:
    """
    Persistent memory store for AI agents.

    Zero mandatory dependencies — uses Python's built-in sqlite3.
    Supports multiple agents and users in the same database.

    Example:
        mem = MemoryStore("./my_agent.db")

        # Add memories
        mem.add("user prefers short responses", tags=["preference"])
        mem.add("user works in healthcare", memory_type=MemoryType.SEMANTIC)

        # Retrieve context for a prompt
        context = mem.context_for("communication preferences")
        # → "- user prefers short responses (importance: 0.50)\\n..."

        # Search
        results = mem.search("response format")
        for r in results:
            print(f"{r.relevance_score:.2f}  {r.content}")

        # Privacy: forget everything for a user
        mem.forget(user_id="user_123")

        # Audit trail
        for event in mem.audit_log():
            print(event)
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        default_agent_id: str = "default",
        default_ttl_days: int | None = None,
        max_memories: int | None = None,
    ) -> None:
        """
        Args:
            path:             Path to SQLite database file, or ':memory:' for in-memory
            default_agent_id: Default agent_id used when none is specified
            default_ttl_days: Default TTL in days for new memories (None = never expire)
            max_memories:     Maximum memories to keep per agent (oldest pruned first)
        """
        self.path = str(path)
        self.default_agent_id = default_agent_id
        self.default_ttl_days = default_ttl_days
        self.max_memories = max_memories
        # For :memory: databases, keep a persistent connection so tables survive
        self._persistent_conn: sqlite3.Connection | None = None
        if self.path == ":memory:":
            self._persistent_conn = sqlite3.connect(
                ":memory:", check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        agent_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        ttl_days: int | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content:     The memory content (plain text)
            memory_type: SEMANTIC / EPISODIC / PROCEDURAL / WORKING
            agent_id:    Agent that owns this memory
            user_id:     User this memory belongs to
            tags:        List of string tags for filtering
            importance:  0.0–1.0 weight (higher = retrieved more often)
            ttl_days:    Days until expiry (overrides default_ttl_days)
            metadata:    Additional key-value data

        Returns:
            The stored Memory object
        """
        agent_id = agent_id or self.default_agent_id
        tags = tags or []
        metadata = metadata or {}
        importance = max(0.0, min(1.0, importance))

        # Resolve TTL
        effective_ttl = ttl_days if ttl_days is not None else self.default_ttl_days
        expires_at = None
        if effective_ttl is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(days=effective_ttl)

        memory = Memory(
            content=content,
            memory_type=memory_type,
            agent_id=agent_id,
            user_id=user_id,
            tags=tags,
            importance=importance,
            expires_at=expires_at,
            metadata=metadata,
        )

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, content, memory_type, agent_id, user_id, tags,
                     importance, created_at, accessed_at, expires_at,
                     access_count, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.memory_type.value,
                    memory.agent_id,
                    memory.user_id,
                    json.dumps(memory.tags),
                    memory.importance,
                    _dt_str(memory.created_at),
                    None,
                    _dt_str(memory.expires_at) if memory.expires_at else None,
                    0,
                    json.dumps(memory.metadata),
                ),
            )
            self._write_audit(
                conn, AuditAction.ADD, agent_id, memory.id,
                f"type={memory_type.value} tags={tags}",
                user_id=user_id,
            )

        if self.max_memories:
            self._prune_if_needed(agent_id)

        return memory

    def search(
        self,
        query: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_expired: bool = False,
    ) -> list[SearchResult]:
        """
        Search memories by relevance to a query string.

        Relevance combines token overlap with importance weighting.
        Override _score_relevance() to plug in vector search.

        Args:
            query:           Natural language search query
            agent_id:        Filter to specific agent (default: all agents)
            user_id:         Filter to specific user
            tags:            Filter to memories with ALL these tags
            memory_type:     Filter to specific memory type
            limit:           Maximum results to return
            min_importance:  Minimum importance threshold
            include_expired: Include expired memories in results

        Returns:
            List of SearchResult ordered by descending relevance score
        """
        agent_id = agent_id or self.default_agent_id
        memories = self._load_memories(
            agent_id=agent_id,
            user_id=user_id,
            tags=tags,
            memory_type=memory_type,
            min_importance=min_importance,
            include_expired=include_expired,
        )

        scored = [
            SearchResult(memory=m, relevance_score=self._score_relevance(query, m))
            for m in memories
        ]
        scored = [r for r in scored if r.relevance_score > 0.0]
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        results = scored[:limit]

        # Update access counts for returned memories
        if results:
            ids = [r.id for r in results]
            with self._conn() as conn:
                for mid in ids:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
                        (_dt_str(datetime.now(timezone.utc)), mid),
                    )
                self._write_audit(
                    conn, AuditAction.SEARCH, agent_id, None,
                    f"query={query[:80]!r} hits={len(results)}",
                    user_id=user_id,
                )

        return results

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_memory(row) if row else None

    def context_for(
        self,
        query: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
        format: str = "bullet",
    ) -> str:
        """
        Return a formatted context string ready to inject into an LLM prompt.

        Args:
            query:    What the agent needs context about
            agent_id: Agent to retrieve memories for
            user_id:  User to retrieve memories for
            limit:    Maximum memories to include
            format:   'bullet' for bullet list, 'numbered' for numbered list,
                      'plain' for plain text concatenation

        Returns:
            Formatted string of relevant memories, or empty string if none found

        Example:
            context = mem.context_for("user preferences")
            # Use in system prompt:
            system = f"You are a helpful assistant.\\n\\nWhat I know:\\n{context}"
        """
        results = self.search(query, agent_id=agent_id, user_id=user_id, limit=limit)
        if not results:
            return ""

        if format == "bullet":
            lines = [f"- {r.content}" for r in results]
        elif format == "numbered":
            lines = [f"{i+1}. {r.content}" for i, r in enumerate(results)]
        else:
            lines = [r.content for r in results]

        return "\n".join(lines)

    def forget(
        self,
        memory_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        older_than_days: int | None = None,
    ) -> int:
        """
        Delete memories matching the given criteria.
        At least one filter must be provided.

        Args:
            memory_id:       Delete a specific memory by ID
            agent_id:        Delete all memories for this agent
            user_id:         Delete all memories for this user
            tags:            Delete memories with ANY of these tags
            older_than_days: Delete memories older than N days

        Returns:
            Number of memories deleted

        Example:
            mem.forget(user_id="user_123")           # right to erasure
            mem.forget(older_than_days=90)           # data minimisation
            mem.forget(tags=["sensitive"])           # category-based deletion
            mem.forget(memory_id="mem_abc")          # specific memory
        """
        if all(v is None for v in [memory_id, agent_id, user_id, tags, older_than_days]):
            raise ValueError(
                "forget() requires at least one filter. "
                "Call forget(agent_id='...') to forget all memories for an agent."
            )

        conditions = []
        params: list = []
        audit_agent = agent_id or self.default_agent_id
        audit_details = []

        if memory_id:
            conditions.append("id = ?")
            params.append(memory_id)
            audit_details.append(f"id={memory_id}")

        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
            audit_details.append(f"agent_id={agent_id}")

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
            audit_details.append(f"user_id={user_id}")

        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            conditions.append("created_at < ?")
            params.append(_dt_str(cutoff))
            audit_details.append(f"older_than={older_than_days}d")

        where = " AND ".join(conditions) if conditions else "1=1"

        # Handle tags filter separately (JSON array in SQLite)
        tag_deleted = 0
        if tags:
            # Load candidates matching other conditions, then filter by tag
            with self._conn() as conn:
                candidates = conn.execute(
                    f"SELECT id, tags, agent_id FROM memories WHERE {where}", params
                ).fetchall()
            tag_ids = [
                row[0] for row in candidates
                if any(t in json.loads(row[1]) for t in tags)
            ]
            if tag_ids:
                placeholders = ",".join("?" * len(tag_ids))
                with self._conn() as conn:
                    cursor = conn.execute(
                        f"DELETE FROM memories WHERE id IN ({placeholders})", tag_ids
                    )
                    tag_deleted = cursor.rowcount
                    self._write_audit(
                        conn, AuditAction.FORGET, audit_agent, None,
                        f"tags={tags} deleted={tag_deleted}",
                    )
            return tag_deleted

        with self._conn() as conn:
            cursor = conn.execute(f"DELETE FROM memories WHERE {where}", params)
            deleted = cursor.rowcount
            self._write_audit(
                conn, AuditAction.FORGET, audit_agent, memory_id,
                " ".join(audit_details) + f" deleted={deleted}",
            )

        return deleted

    def expire_stale(self) -> int:
        """
        Delete all memories that have passed their expires_at timestamp.
        Call periodically or at agent startup.

        Returns:
            Number of memories deleted
        """
        now = _dt_str(datetime.now(timezone.utc))
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            deleted = cursor.rowcount
            if deleted:
                self._write_audit(
                    conn, AuditAction.EXPIRE, self.default_agent_id, None,
                    f"expired={deleted}",
                )
        return deleted

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Memory | None:
        """Update fields on an existing memory."""
        memory = self.get(memory_id)
        if memory is None:
            return None

        updates: list[str] = []
        params: list = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if importance is not None:
            updates.append("importance = ?")
            params.append(max(0.0, min(1.0, importance)))
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return memory

        params.append(memory_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params
            )
            self._write_audit(
                conn, AuditAction.UPDATE, memory.agent_id, memory_id,
                f"fields={[u.split(' ')[0] for u in updates]}",
            )

        return self.get(memory_id)

    def audit_log(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """
        Return audit log entries, most recent first.

        The audit log is append-only — entries are never deleted.
        It records every add, search, forget, update, and expire operation.
        """
        conditions = []
        params: list = []

        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("action = ?")
            params.append(action.value)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()

        return [_row_to_audit(row) for row in rows]

    def stats(self, agent_id: str | None = None) -> dict:
        """Return summary statistics for the memory store."""
        agent_id = agent_id or self.default_agent_id
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories WHERE agent_id = ? GROUP BY memory_type",
                (agent_id,),
            ).fetchall()
            avg_importance = conn.execute(
                "SELECT AVG(importance) FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchone()[0]
            expiring_soon = conn.execute(
                """SELECT COUNT(*) FROM memories WHERE agent_id = ?
                   AND expires_at IS NOT NULL
                   AND expires_at < datetime('now', '+7 days')""",
                (agent_id,),
            ).fetchone()[0]
            audit_total = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE agent_id = ?", (agent_id,)
            ).fetchone()[0]

        return {
            "agent_id": agent_id,
            "total_memories": total,
            "by_type": {row[0]: row[1] for row in by_type},
            "avg_importance": round(avg_importance or 0.0, 3),
            "expiring_within_7_days": expiring_soon,
            "audit_events": audit_total,
        }

    def all_memories(
        self,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[Memory]:
        """Return all memories for an agent (for inspection / export)."""
        agent_id = agent_id or self.default_agent_id
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    # ── Relevance scoring (override to plug in vector search) ─────────────────

    def _score_relevance(self, query: str, memory: Memory) -> float:
        """
        Default keyword-based relevance scorer.
        Combined score: token overlap + importance weighting.

        Override this method to use semantic/vector search:

            from agent_memory import MemoryStore
            from sentence_transformers import SentenceTransformer
            import numpy as np

            class SemanticMemoryStore(MemoryStore):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')

                def _score_relevance(self, query, memory):
                    q_emb = self.model.encode(query)
                    m_emb = self.model.encode(memory.content)
                    cos_sim = float(np.dot(q_emb, m_emb) /
                                    (np.linalg.norm(q_emb) * np.linalg.norm(m_emb)))
                    return cos_sim * 0.7 + memory.importance * 0.3
        """
        query_tokens = set(re.findall(r'\w+', query.lower()))
        content_tokens = set(re.findall(r'\w+', memory.content.lower()))

        if not query_tokens:
            return 0.0

        # Jaccard-like overlap
        overlap = len(query_tokens & content_tokens) / len(query_tokens)

        # Importance only contributes when there is some token overlap
        if overlap == 0.0:
            return 0.0
        score = overlap * 0.7 + memory.importance * 0.3

        # Boost for tag matches
        tag_tokens = set(t.lower() for t in memory.tags)
        if query_tokens & tag_tokens:
            score = min(1.0, score + 0.1)

        return round(score, 4)

    # ── Database internals ────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'semantic',
                    agent_id TEXT NOT NULL DEFAULT 'default',
                    user_id TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    user_id TEXT,
                    memory_id TEXT,
                    timestamp TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON memories(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id)")

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if self._persistent_conn is not None:
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _load_memories(
        self,
        agent_id: str,
        user_id: str | None,
        tags: list[str] | None,
        memory_type: MemoryType | None,
        min_importance: float,
        include_expired: bool,
    ) -> list[Memory]:
        conditions = ["agent_id = ?"]
        params: list = [agent_id]

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        if min_importance > 0.0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        if not include_expired:
            conditions.append(
                "(expires_at IS NULL OR expires_at > datetime('now'))"
            )

        where = " AND ".join(conditions)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY importance DESC",
                params,
            ).fetchall()

        memories = [_row_to_memory(row) for row in rows]

        # Further filter by tags if specified
        if tags:
            memories = [m for m in memories if all(t in m.tags for t in tags)]

        return memories

    def _write_audit(
        self,
        conn: sqlite3.Connection,
        action: AuditAction,
        agent_id: str,
        memory_id: str | None,
        details: str,
        user_id: str | None = None,
    ) -> None:
        import uuid as _uuid
        event_id = f"audit_{_uuid.uuid4().hex[:12]}"
        conn.execute(
            """INSERT INTO audit_log (id, action, agent_id, user_id, memory_id, timestamp, details)
               VALUES (?,?,?,?,?,?,?)""",
            (
                event_id,
                action.value,
                agent_id,
                user_id,
                memory_id,
                _dt_str(datetime.now(timezone.utc)),
                details,
            ),
        )

    def _prune_if_needed(self, agent_id: str) -> None:
        """Prune oldest, least-important memories if max_memories exceeded."""
        if not self.max_memories:
            return
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ?", (agent_id,)
            ).fetchone()[0]
            if count > self.max_memories:
                excess = count - self.max_memories
                conn.execute(
                    """DELETE FROM memories WHERE id IN (
                        SELECT id FROM memories WHERE agent_id = ?
                        ORDER BY importance ASC, created_at ASC LIMIT ?
                    )""",
                    (agent_id, excess),
                )


# ── Helper functions ──────────────────────────────────────────────────────────

def _dt_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        tags=json.loads(row["tags"] or "[]"),
        importance=row["importance"],
        created_at=_parse_dt(row["created_at"]) or datetime.now(timezone.utc),
        accessed_at=_parse_dt(row["accessed_at"]),
        expires_at=_parse_dt(row["expires_at"]),
        access_count=row["access_count"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _row_to_audit(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        id=row["id"],
        action=AuditAction(row["action"]),
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        memory_id=row["memory_id"],
        timestamp=_parse_dt(row["timestamp"]) or datetime.now(timezone.utc),
        details=row["details"],
    )
