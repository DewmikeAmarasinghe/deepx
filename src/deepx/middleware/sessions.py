from __future__ import annotations

from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession


def create_session(session_id: str, db_path: str = ":memory:"):
    """Create a session backed by SQLite with automatic server-side compaction.

    Args:
        session_id: Unique identifier for this conversation thread.
        db_path: Path to the SQLite database file.  Defaults to ``:memory:``
            (ephemeral, lost when the process exits).  Pass a file path such
            as ``"agent.db"`` for persistence across runs.

    Compaction is handled automatically by ``OpenAIResponsesCompactionSession``:
    once the stored history reaches 10 items the SDK calls ``responses.compact``
    server-side using ``gpt-4.1`` and replaces the history with the summary.
    No manual configuration is needed.
    """
    raw = SQLiteSession(session_id, db_path)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )