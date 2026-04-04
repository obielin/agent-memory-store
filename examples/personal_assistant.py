"""
Example: Personal assistant with persistent memory.
Shows how to add agent-memory-store to any existing Claude agent in 3 lines.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/personal_assistant.py
"""

from agent_memory import MemoryStore, MemoryType

# ── 1. Initialise memory (3 lines to add memory to any agent) ─────────────────
mem = MemoryStore("assistant.db", default_agent_id="personal_assistant")
mem.expire_stale()  # clean up expired working memories at session start


def chat(user_message: str, user_id: str = "user_001") -> str:
    """A simple agent turn with memory injection and storage."""

    # ── 2. Retrieve relevant context ──────────────────────────────────────────
    context = mem.context_for(user_message, user_id=user_id, limit=5)

    # ── 3. Build prompt with memory ───────────────────────────────────────────
    system = "You are a helpful personal assistant."
    if context:
        system += f"\n\nWhat you know about this user:\n{context}"

    print(f"\n[Context injected]:\n{context or '(none yet)'}\n")

    # ── 4. Call Claude ─────────────────────────────────────────────────────────
    try:
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        assistant_reply = response.content[0].text
    except ImportError:
        assistant_reply = f"[Demo mode — no API key] I would respond to: {user_message}"

    # ── 5. Store what was learned from this turn ───────────────────────────────
    _extract_and_store_memories(user_message, user_id)

    return assistant_reply


def _extract_and_store_memories(message: str, user_id: str) -> None:
    """Simple heuristic to store facts mentioned by the user."""
    msg_lower = message.lower()

    if "i prefer" in msg_lower or "i like" in msg_lower or "i want" in msg_lower:
        mem.add(
            f"User preference: {message[:100]}",
            memory_type=MemoryType.PROCEDURAL,
            user_id=user_id,
            tags=["preference"],
            importance=0.8,
        )

    if "i work" in msg_lower or "my job" in msg_lower or "my role" in msg_lower:
        mem.add(
            f"User background: {message[:100]}",
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            tags=["background"],
            importance=0.9,
        )

    # Store the conversation turn as episodic memory (expires in 7 days)
    mem.add(
        f"User said: {message[:80]}",
        memory_type=MemoryType.EPISODIC,
        user_id=user_id,
        tags=["conversation"],
        importance=0.4,
        ttl_days=7,
    )


def show_memory_stats() -> None:
    stats = mem.stats()
    print(f"\n[Memory stats]")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Avg importance: {stats['avg_importance']}")


if __name__ == "__main__":
    print("Personal Assistant with Persistent Memory")
    print("=" * 50)
    print("Type 'quit' to exit, 'stats' to see memory stats, 'audit' for audit log\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            show_memory_stats()
            continue
        if user_input.lower() == "audit":
            for event in mem.audit_log(limit=10):
                print(f"  {event}")
            continue

        reply = chat(user_input)
        print(f"\nAssistant: {reply}\n")
