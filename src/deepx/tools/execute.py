from __future__ import annotations

import asyncio

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Run a **host shell** command (not the virtual file-tool layer).

    - **Working directory** is the backend’s **project root** (the same tree file tools use:
      paths like ``/test_demo/foo`` in file tools map to ``<root>/test_demo/foo``).
    - Shell commands are passed to the host shell as given; prefer file tools for project files.
    - Requires a backend that implements ``execute`` (e.g. ``LocalShellBackend``). Output is
      capped. Use short, non-interactive commands only.
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    cmd = (command or "").strip()
    if not cmd:
        return "No command provided."
    return await asyncio.to_thread(b.execute, sid, cmd)
