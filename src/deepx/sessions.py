from __future__ import annotations

from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession


def create_session(session_id: str, db_path: str = ":memory:"):
    resolved = db_path.strip() or ":memory:"
    raw = SQLiteSession(session_id, resolved)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )
