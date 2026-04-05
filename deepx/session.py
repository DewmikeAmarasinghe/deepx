from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession


def session_factory(session_id: str, db_path: str):
    raw = SQLiteSession(session_id, db_path)
    return OpenAIResponsesCompactionSession(session_id=session_id, underlying_session=raw)
