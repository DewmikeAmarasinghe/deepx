"""Resume Chainlit threads with saved session preferences."""

from __future__ import annotations

import chainlit as cl
from chainlit.types import ThreadDict


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    tid = thread.get("id")
    if tid:
        cl.user_session.set("session_id", str(tid))
    meta = thread.get("metadata")
    if isinstance(meta, dict) and "use_temporal" in meta:
        cl.user_session.set("use_temporal", bool(meta["use_temporal"]))
