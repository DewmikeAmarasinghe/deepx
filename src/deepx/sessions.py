from __future__ import annotations

from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession


def create_session(session_id: str, checkpointer: str = ":memory:"):
    """Conversation history for the agents SDK.

    ``checkpointer`` is a SQLite database path (including ``\":memory:\"`` for an in-memory DB)
    passed to :class:`~agents.memory.SQLiteSession`, wrapped in
    :class:`~agents.memory.OpenAIResponsesCompactionSession`.
    """
    resolved = checkpointer.strip() or ":memory:"
    raw = SQLiteSession(session_id, resolved)

    def should_compact_at_90_percent(context):
        usage_data = context.get("usage", {})
        total_tokens = usage_data.get("total_tokens", 0)
        limit = context.get("model_context_window", 1)
        return total_tokens >= (limit * 0.9)

    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        model="gpt-5-nano",
        underlying_session=raw,
        should_trigger_compaction=should_compact_at_90_percent,
    )


__all__ = ["create_session"]
