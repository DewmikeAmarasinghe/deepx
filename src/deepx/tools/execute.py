from __future__ import annotations

import asyncio

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext

EXECUTE_TIMEOUT_CAP_S = 600.0


@function_tool
async def execute(
    ctx: RunContextWrapper[AgentContext],
    command: str,
    timeout_seconds: float = 120.0,
) -> str:
    """Run one **host shell** command in the backend’s project root (same tree as file tools).

    Pass **`command`**: a non-interactive shell string. **`timeout_seconds`** is clamped. Unsupported
    backends return a clear error. Very large stdout/stderr may be spilled to
    **`/_outputs/large_tool_results/`** by the framework — use **`read_file`** on the path from the
    tool message if so.
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    cmd = str(command).strip()
    if not cmd:
        return "No command provided. Pass a non-empty `command` string."
    t = float(timeout_seconds)
    t = min(max(t, 1.0), EXECUTE_TIMEOUT_CAP_S)

    def run() -> str:
        return b.execute(sid, cmd, timeout=t)

    return await asyncio.to_thread(run)
