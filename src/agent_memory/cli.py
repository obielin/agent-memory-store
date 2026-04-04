"""
agent-memory CLI — inspect and manage memory stores from the terminal.

Usage:
    agent-memory stats agent.db
    agent-memory list agent.db
    agent-memory search agent.db "user preferences"
    agent-memory forget agent.db --older-than 30
    agent-memory audit agent.db
    agent-memory export agent.db memories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-memory",
        description="Inspect and manage agent memory stores.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    p_stats = sub.add_parser("stats", help="Show memory store statistics")
    p_stats.add_argument("db", help="Path to SQLite database")
    p_stats.add_argument("--agent", default=None, help="Agent ID to filter")

    # list
    p_list = sub.add_parser("list", help="List stored memories")
    p_list.add_argument("db", help="Path to SQLite database")
    p_list.add_argument("--agent", default=None, help="Agent ID to filter")
    p_list.add_argument("--limit", type=int, default=20)

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("db", help="Path to SQLite database")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--agent", default=None)
    p_search.add_argument("--limit", type=int, default=10)

    # forget
    p_forget = sub.add_parser("forget", help="Delete memories")
    p_forget.add_argument("db", help="Path to SQLite database")
    p_forget.add_argument("--agent", default=None, help="Delete all memories for agent")
    p_forget.add_argument("--user", default=None, help="Delete all memories for user")
    p_forget.add_argument("--older-than", type=int, default=None, metavar="DAYS",
                          help="Delete memories older than N days")
    p_forget.add_argument("--id", default=None, help="Delete specific memory by ID")

    # audit
    p_audit = sub.add_parser("audit", help="Show audit log")
    p_audit.add_argument("db", help="Path to SQLite database")
    p_audit.add_argument("--agent", default=None)
    p_audit.add_argument("--limit", type=int, default=20)

    # export
    p_export = sub.add_parser("export", help="Export memories to JSON")
    p_export.add_argument("db", help="Path to SQLite database")
    p_export.add_argument("output", help="Output JSON file")
    p_export.add_argument("--agent", default=None)

    args = parser.parse_args()

    # Import here to keep CLI startup fast
    from agent_memory.store import MemoryStore

    db_path = args.db
    if not Path(db_path).exists() and args.command != "stats":
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    mem = MemoryStore(db_path, default_agent_id=args.agent or "default")

    if args.command == "stats":
        stats = mem.stats(agent_id=args.agent)
        print(f"\nMemory Store: {db_path}")
        print(f"Agent:        {stats['agent_id']}")
        print(f"Total:        {stats['total_memories']} memories")
        print(f"Avg import.:  {stats['avg_importance']:.3f}")
        print(f"Expiring 7d:  {stats['expiring_within_7_days']}")
        print(f"Audit events: {stats['audit_events']}")
        if stats["by_type"]:
            print("\nBy type:")
            for t, count in stats["by_type"].items():
                print(f"  {t:<12} {count}")

    elif args.command == "list":
        memories = mem.all_memories(agent_id=args.agent, limit=args.limit)
        print(f"\n{len(memories)} memories (agent={args.agent or 'all'}):\n")
        for m in memories:
            tags = f" [{', '.join(m.tags)}]" if m.tags else ""
            exp = f" expires={m.expires_at.date()}" if m.expires_at else ""
            print(f"  {m.id}  [{m.memory_type.value:<10}]  imp={m.importance:.2f}{tags}{exp}")
            print(f"    {m.content[:100]}")

    elif args.command == "search":
        results = mem.search(args.query, agent_id=args.agent, limit=args.limit)
        print(f"\nSearch: {args.query!r}  ({len(results)} results)\n")
        for r in results:
            print(f"  [{r.relevance_score:.3f}]  {r.id}  {r.content[:100]}")

    elif args.command == "forget":
        if not any([args.agent, args.user, args.older_than, args.id]):
            print("Specify at least one filter: --agent, --user, --older-than, --id")
            sys.exit(1)
        deleted = mem.forget(
            memory_id=args.id,
            agent_id=args.agent,
            user_id=args.user,
            older_than_days=args.older_than,
        )
        print(f"Deleted {deleted} memories.")

    elif args.command == "audit":
        events = mem.audit_log(agent_id=args.agent, limit=args.limit)
        print(f"\nAudit log ({len(events)} events):\n")
        for e in events:
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            mid = f"  mem={e.memory_id}" if e.memory_id else ""
            print(f"  {ts}  {e.action.value:<10}  agent={e.agent_id}{mid}  {e.details}")

    elif args.command == "export":
        memories = mem.all_memories(agent_id=args.agent)
        data = [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type.value,
                "agent_id": m.agent_id,
                "user_id": m.user_id,
                "tags": m.tags,
                "importance": m.importance,
                "created_at": m.created_at.isoformat(),
                "metadata": m.metadata,
            }
            for m in memories
        ]
        Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Exported {len(data)} memories to {args.output}")


if __name__ == "__main__":
    main()
