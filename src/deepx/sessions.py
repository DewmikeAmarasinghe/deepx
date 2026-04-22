from __future__ import annotations

from agents.items import TResponseInputItem
from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession
from agents.memory.session import SessionABC
from agents.memory.session_settings import SessionSettings, resolve_session_limit


def is_memory_checkpointer(checkpointer: str) -> bool:
    """True when history should use :class:`AsyncListSession` (no SQLite, no thread pool).

    Accepts the sentinel ``\"memory\"`` or legacy SQLite in-memory URI ``\":memory:\"``.
    """
    s = checkpointer.strip().lower()
    return s in ("memory", ":memory:")


class AsyncListSession(SessionABC):
    """In-process async session store (no threads, no SQLite).

    Used when :func:`is_memory_checkpointer` is true for ``checkpointer`` (e.g. durable
    workflow sandboxes that cannot run ``asyncio.to_thread`` for SQLite).
    """

    def __init__(
        self,
        session_id: str,
        *,
        session_settings: SessionSettings | None = None,
    ) -> None:
        self.session_id = session_id
        self.session_settings = session_settings
        self._items: list[TResponseInputItem] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        session_limit = resolve_session_limit(limit, self.session_settings)
        if session_limit is None:
            return list(self._items)
        if session_limit <= 0:
            return []
        return list(self._items[-session_limit:])

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if items:
            self._items.extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        if not self._items:
            return None
        return self._items.pop()

    async def clear_session(self) -> None:
        self._items.clear()


def create_session(session_id: str, checkpointer: str = "memory"):
    """Conversation history for the agents SDK.

    - ``checkpointer`` is ``\"memory\"`` or ``\":memory:\"`` → :class:`AsyncListSession`.
    - Otherwise → :class:`~agents.memory.SQLiteSession` at the given filesystem path.
    """
    resolved = checkpointer.strip()
    if not resolved:
        resolved = "memory"
    if is_memory_checkpointer(resolved):
        raw: SessionABC = AsyncListSession(session_id)
    else:
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


__all__ = ["AsyncListSession", "create_session", "is_memory_checkpointer"]
