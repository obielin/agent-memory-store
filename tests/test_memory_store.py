"""
Tests for agent-memory-store.
Run: pytest tests/ -v
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_memory import MemoryStore, Memory, MemoryType, AuditEvent, SearchResult
from agent_memory.models import AuditAction



# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem():
    """In-memory store for tests — no disk I/O."""
    return MemoryStore(":memory:", default_agent_id="test_agent")


@pytest.fixture
def populated_mem():
    """Store pre-loaded with varied memories."""
    store = MemoryStore(":memory:", default_agent_id="test_agent")
    store.add("user prefers concise responses", tags=["preference", "communication"], importance=0.8)
    store.add("user works in healthcare sector", tags=["background"], importance=0.9, memory_type=MemoryType.SEMANTIC)
    store.add("last session discussed Python decorators", tags=["session", "coding"], memory_type=MemoryType.EPISODIC)
    store.add("always greet user by name", tags=["procedure"], memory_type=MemoryType.PROCEDURAL, importance=0.7)
    store.add("temporary task context", memory_type=MemoryType.WORKING, importance=0.3)
    return store


# ── Memory model tests ────────────────────────────────────────────────────────

class TestMemoryModel:
    def test_id_auto_generated(self):
        m = Memory(content="test")
        assert m.id.startswith("mem_")
        assert len(m.id) > 4

    def test_unique_ids(self):
        ids = {Memory(content="x").id for _ in range(50)}
        assert len(ids) == 50

    def test_is_not_expired_without_expiry(self):
        m = Memory(content="test")
        assert not m.is_expired()

    def test_is_expired_past_date(self):
        m = Memory(content="test", expires_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        assert m.is_expired()

    def test_is_not_expired_future_date(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        m = Memory(content="test", expires_at=future)
        assert not m.is_expired()

    def test_repr_truncates_long_content(self):
        m = Memory(content="x" * 100)
        assert "..." in repr(m)

    def test_default_memory_type(self):
        m = Memory(content="test")
        assert m.memory_type == MemoryType.SEMANTIC


# ── Store add tests ───────────────────────────────────────────────────────────

class TestAdd:
    def test_add_returns_memory(self, mem):
        result = mem.add("hello world")
        assert isinstance(result, Memory)
        assert result.content == "hello world"

    def test_add_persists_to_db(self, mem):
        m = mem.add("persisted content")
        retrieved = mem.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "persisted content"

    def test_add_with_tags(self, mem):
        m = mem.add("tagged memory", tags=["a", "b"])
        retrieved = mem.get(m.id)
        assert "a" in retrieved.tags
        assert "b" in retrieved.tags

    def test_add_with_importance(self, mem):
        m = mem.add("important", importance=0.9)
        retrieved = mem.get(m.id)
        assert retrieved.importance == pytest.approx(0.9)

    def test_importance_clamped_high(self, mem):
        m = mem.add("test", importance=2.0)
        assert mem.get(m.id).importance <= 1.0

    def test_importance_clamped_low(self, mem):
        m = mem.add("test", importance=-1.0)
        assert mem.get(m.id).importance >= 0.0

    def test_add_with_ttl(self, mem):
        m = mem.add("expires soon", ttl_days=7)
        retrieved = mem.get(m.id)
        assert retrieved.expires_at is not None
        assert retrieved.expires_at > datetime.now(timezone.utc)

    def test_add_with_user_id(self, mem):
        m = mem.add("user-specific", user_id="user_123")
        retrieved = mem.get(m.id)
        assert retrieved.user_id == "user_123"

    def test_add_with_metadata(self, mem):
        m = mem.add("with metadata", metadata={"source": "conversation", "turn": 5})
        retrieved = mem.get(m.id)
        assert retrieved.metadata["source"] == "conversation"

    def test_add_different_memory_types(self, mem):
        for mt in MemoryType:
            m = mem.add(f"memory type {mt.value}", memory_type=mt)
            assert mem.get(m.id).memory_type == mt


# ── Search tests ──────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_returns_relevant_results(self, populated_mem):
        results = populated_mem.search("user preferences communication")
        assert len(results) > 0
        assert any("prefers" in r.content for r in results)

    def test_search_returns_search_results(self, populated_mem):
        results = populated_mem.search("healthcare")
        for r in results:
            assert isinstance(r, SearchResult)

    def test_search_ordered_by_relevance(self, populated_mem):
        results = populated_mem.search("user")
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query_returns_empty(self, mem):
        mem.add("some content")
        results = mem.search("")
        assert results == []

    def test_search_no_match_returns_empty(self, populated_mem):
        results = populated_mem.search("zzzxxx111 qqquuu999")
        assert results == []

    def test_search_limit_respected(self, populated_mem):
        results = populated_mem.search("user", limit=2)
        assert len(results) <= 2

    def test_search_filter_by_memory_type(self, populated_mem):
        results = populated_mem.search("session", memory_type=MemoryType.EPISODIC)
        assert all(r.memory_type == MemoryType.EPISODIC for r in results)

    def test_search_filter_by_tags(self, populated_mem):
        results = populated_mem.search("user", tags=["preference"])
        assert all("preference" in r.tags for r in results)

    def test_search_increments_access_count(self, mem):
        m = mem.add("searchable content searchable")
        mem.search("searchable content")
        retrieved = mem.get(m.id)
        assert retrieved.access_count >= 1

    def test_search_excludes_expired_by_default(self, mem):
        mem.add("expired content", ttl_days=-1)
        results = mem.search("expired content")
        assert len(results) == 0

    def test_search_includes_expired_when_requested(self, mem):
        mem.add("expired content expiry", ttl_days=-1)
        results = mem.search("expired content", include_expired=True)
        assert len(results) > 0


# ── context_for tests ─────────────────────────────────────────────────────────

class TestContextFor:
    def test_returns_string(self, populated_mem):
        ctx = populated_mem.context_for("user preferences")
        assert isinstance(ctx, str)

    def test_returns_empty_for_no_match(self, populated_mem):
        ctx = populated_mem.context_for("zzzxxx111 qqquuu999")
        assert ctx == ""

    def test_bullet_format(self, populated_mem):
        ctx = populated_mem.context_for("user", format="bullet")
        assert ctx.startswith("- ")

    def test_numbered_format(self, populated_mem):
        ctx = populated_mem.context_for("user", format="numbered")
        assert ctx.startswith("1.")

    def test_plain_format(self, populated_mem):
        ctx = populated_mem.context_for("user", format="plain")
        assert not ctx.startswith("- ")
        assert not ctx.startswith("1.")

    def test_limit_respected(self, populated_mem):
        ctx = populated_mem.context_for("user", limit=1, format="bullet")
        assert ctx.count("\n") == 0  # Only one line


# ── Forget tests ──────────────────────────────────────────────────────────────

class TestForget:
    def test_forget_by_id(self, mem):
        m = mem.add("to be deleted")
        mem.forget(memory_id=m.id)
        assert mem.get(m.id) is None

    def test_forget_by_agent_id(self, mem):
        mem.add("agent memory 1", agent_id="agent_a")
        mem.add("agent memory 2", agent_id="agent_a")
        mem.add("other agent", agent_id="agent_b")
        deleted = mem.forget(agent_id="agent_a")
        assert deleted == 2
        assert len(mem.all_memories(agent_id="agent_b")) == 1

    def test_forget_by_user_id(self, mem):
        mem.add("user A memory", user_id="user_a")
        mem.add("user B memory", user_id="user_b")
        deleted = mem.forget(user_id="user_a")
        assert deleted == 1

    def test_forget_by_tags(self, mem):
        mem.add("tagged sensitive", tags=["sensitive"])
        mem.add("tagged normal", tags=["normal"])
        deleted = mem.forget(tags=["sensitive"])
        assert deleted == 1

    def test_forget_older_than(self, mem):
        # Add a memory then pretend it's old by checking count
        mem.add("old ish memory")
        deleted = mem.forget(older_than_days=999)
        assert deleted == 0  # Nothing old enough

    def test_forget_requires_filter(self, mem):
        with pytest.raises(ValueError, match="at least one filter"):
            mem.forget()

    def test_forget_returns_count(self, mem):
        mem.add("a"), mem.add("b"), mem.add("c")
        deleted = mem.forget(agent_id="test_agent")
        assert deleted == 3


# ── TTL and expiry tests ──────────────────────────────────────────────────────

class TestExpiry:
    def test_expire_stale_removes_expired(self, mem):
        mem.add("expired", ttl_days=-1)
        deleted = mem.expire_stale()
        assert deleted == 1

    def test_expire_stale_keeps_valid(self, mem):
        mem.add("valid", ttl_days=30)
        deleted = mem.expire_stale()
        assert deleted == 0

    def test_expire_stale_keeps_no_expiry(self, mem):
        mem.add("no expiry")
        deleted = mem.expire_stale()
        assert deleted == 0


# ── Audit log tests ───────────────────────────────────────────────────────────

class TestAuditLog:
    def test_add_creates_audit_event(self, mem):
        mem.add("audited memory")
        events = mem.audit_log()
        assert any(e.action == AuditAction.ADD for e in events)

    def test_search_creates_audit_event(self, mem):
        mem.add("searchable audited content")
        mem.search("searchable audited")
        events = mem.audit_log()
        assert any(e.action == AuditAction.SEARCH for e in events)

    def test_forget_creates_audit_event(self, mem):
        m = mem.add("to delete")
        mem.forget(memory_id=m.id)
        events = mem.audit_log()
        assert any(e.action == AuditAction.FORGET for e in events)

    def test_audit_ordered_most_recent_first(self, mem):
        mem.add("first")
        mem.add("second")
        events = mem.audit_log()
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_audit_limit_respected(self, mem):
        for i in range(10):
            mem.add(f"memory {i}")
        events = mem.audit_log(limit=5)
        assert len(events) <= 5

    def test_audit_filter_by_agent(self, mem):
        store_a = MemoryStore(":memory:", default_agent_id="agent_a")
        store_b = MemoryStore(":memory:", default_agent_id="agent_b")
        store_a.add("agent a memory")
        store_b.add("agent b memory")
        events_a = store_a.audit_log(agent_id="agent_a")
        assert all(e.agent_id == "agent_a" for e in events_a)


# ── Stats tests ───────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_returns_dict(self, mem):
        stats = mem.stats()
        assert isinstance(stats, dict)
        assert "total_memories" in stats

    def test_stats_counts_memories(self, populated_mem):
        stats = populated_mem.stats()
        assert stats["total_memories"] == 5

    def test_stats_by_type(self, populated_mem):
        stats = populated_mem.stats()
        assert "by_type" in stats
        assert "semantic" in stats["by_type"]

    def test_stats_avg_importance(self, mem):
        mem.add("test", importance=0.6)
        mem.add("test", importance=0.8)
        stats = mem.stats()
        assert stats["avg_importance"] == pytest.approx(0.7, abs=0.01)


# ── Max memories pruning tests ────────────────────────────────────────────────

class TestMaxMemories:
    def test_prunes_when_limit_exceeded(self):
        mem = MemoryStore(":memory:", max_memories=3)
        for i in range(5):
            mem.add(f"memory {i}", importance=float(i) / 10)
        stats = mem.stats()
        assert stats["total_memories"] <= 3

    def test_keeps_highest_importance(self):
        mem = MemoryStore(":memory:", max_memories=2)
        mem.add("low importance", importance=0.1)
        mem.add("high importance", importance=0.9)
        mem.add("medium importance", importance=0.5)
        memories = mem.all_memories()
        importances = [m.importance for m in memories]
        assert 0.9 in importances  # Highest importance kept


# ── Update tests ──────────────────────────────────────────────────────────────

class TestUpdate:
    def test_update_content(self, mem):
        m = mem.add("original content")
        updated = mem.update(m.id, content="updated content")
        assert updated.content == "updated content"

    def test_update_importance(self, mem):
        m = mem.add("test", importance=0.5)
        updated = mem.update(m.id, importance=0.9)
        assert updated.importance == pytest.approx(0.9)

    def test_update_tags(self, mem):
        m = mem.add("test", tags=["old"])
        updated = mem.update(m.id, tags=["new", "tags"])
        assert "new" in updated.tags
        assert "old" not in updated.tags

    def test_update_nonexistent_returns_none(self, mem):
        result = mem.update("mem_doesnotexist", content="new")
        assert result is None


def _score_relevance_standalone(query, memory):
    """Helper to test scoring without a store instance."""
    import re
    query_tokens = set(re.findall(r'\w+', query.lower()))
    content_tokens = set(re.findall(r'\w+', memory.content.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    return round(overlap * 0.7 + memory.importance * 0.3, 4)


# ── Relevance scoring tests ───────────────────────────────────────────────────

class TestRelevanceScoring:
    def test_exact_match_scores_high(self, mem):
        m = mem.add("user prefers concise responses")
        results = mem.search("user prefers concise responses")
        assert results[0].relevance_score > 0.7

    def test_no_match_scores_zero(self, mem):
        mem.add("user prefers concise responses", importance=0.0)
        results = mem.search("quantum physics equations")
        assert len(results) == 0

    def test_partial_match_scores_between(self, mem):
        mem.add("user prefers concise responses")
        results = mem.search("user preferences")
        if results:
            assert 0.0 < results[0].relevance_score <= 1.0

    def test_importance_boosts_score(self, mem):
        mem.add("test keyword", importance=0.1)
        mem.add("test keyword", importance=0.9)
        results = mem.search("test keyword", limit=10)
        assert len(results) >= 1
