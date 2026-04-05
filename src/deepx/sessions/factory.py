from __future__ import annotations
from agents.memory import SQLiteSession, OpenAIResponsesCompactionSession


def create_session(session_id: str, db_path: str | None = None):
    if db_path is None:
        return SQLiteSession(session_id, ":memory:")

    raw = SQLiteSession(session_id, db_path)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )