"""Push Chat Settings (Temporal switch) when a chat starts."""

from __future__ import annotations

import chainlit as cl
from chainlit.chat_settings import ChatSettings
from chainlit.input_widget import Switch

from test_demo.ui.runs.session_prefs import default_use_temporal


@cl.on_chat_start
async def on_chat_start() -> None:
    if cl.user_session.get("use_temporal") is None:
        cl.user_session.set("use_temporal", default_use_temporal())
    use_t = bool(cl.user_session.get("use_temporal"))
    await ChatSettings(
        [
            Switch(
                id="use_temporal",
                label="Use Temporal for agent runs",
                initial=use_t,
                description="Off: local streaming in this process. On: durable run (worker must be up).",
            )
        ]
    ).send()
