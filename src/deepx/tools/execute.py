from __future__ import annotations

import asyncio

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Run a **host shell** command (not the virtual file-tool layer).

    - **Working directory** is the backend’s **host project root** (e.g. repo root for
      ``LocalShellBackend``).
    - **No path rewriting:** strings like ``/_workspace_/note.md`` inside the command are passed
      to the shell literally—they are **not** mapped to session storage. To touch session files
      from the shell, use the **real on-disk path** (see the FILESYSTEM section in your system
      prompt). Prefer ``read_file`` / ``write_file`` when possible.
    - Requires a backend that implements ``execute`` (e.g. ``LocalShellBackend``). Output is
      capped. Use short, non-interactive commands only.
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    cmd = (command or "").strip()
    if not cmd:
        return "No command provided."
    return await asyncio.to_thread(b.execute, sid, cmd)
