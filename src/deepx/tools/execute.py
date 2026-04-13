from __future__ import annotations

import asyncio

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Run a shell command on the host with working directory set to the backend host root.

    Requires a backend that implements ``execute`` (e.g. ``LocalShellBackend``). Output is capped.
    Prefer short, non-interactive commands.
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    cmd = (command or "").strip()
    if not cmd:
        return "No command provided."
    return await asyncio.to_thread(b.execute, sid, cmd)
