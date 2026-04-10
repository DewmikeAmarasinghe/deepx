from __future__ import annotations

from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession


def create_session(session_id: str, checkpointer: str = ":memory:"):
    """SQLite path for conversation history (agents SDK), or ':memory:' for ephemeral storage."""
    resolved = checkpointer.strip() or ":memory:"
    raw = SQLiteSession(session_id, resolved)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )
