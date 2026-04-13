from __future__ import annotations

import asyncio

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Run a shell command in the session workspace directory (local process, not a container).

    Uses the host shell with working directory set to this session's `/_workspace_/` folder.
    Output is capped. Prefer short, non-interactive commands; set timeouts implicitly via the
    backend (default roughly two minutes).
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    runner = getattr(b, "run_shell_command", None)
    if runner is None:
        return "execute is not available for this backend."
    cmd = (command or "").strip()
    if not cmd:
        return "No command provided."
    return await asyncio.to_thread(lambda: runner(sid, cmd))
