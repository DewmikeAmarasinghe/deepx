"""Chat profiles and settings updates (Temporal toggle + log focus)."""

from __future__ import annotations

import chainlit as cl
from chainlit.context import context
from chainlit.data import get_data_layer


@cl.set_chat_profiles
async def chat_profiles(
    user: cl.User | None = None,
    language: str | None = None,
) -> list[cl.ChatProfile]:
    from test_demo.orchestrator import chat_profile_agent_names

    names = chat_profile_agent_names()
    return [
        cl.ChatProfile(
            name=n,
            markdown_description=f"After each message, emphasise tool JSON logs for **{n}**.",
        )
        for n in names[:12]
    ]


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    prof = settings.get("chat_profile")
    if isinstance(prof, str) and prof.strip():
        cl.user_session.set("preferred_log_agent", prof.strip())
    if "use_temporal" in settings:
        val = bool(settings["use_temporal"])
        cl.user_session.set("use_temporal", val)
        dl = get_data_layer()
        if dl is not None:
            try:
                await dl.update_thread(
                    thread_id=context.session.thread_id,
                    metadata={"use_temporal": val},
                )
            except Exception:
                pass
