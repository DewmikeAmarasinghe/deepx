from __future__ import annotations

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext

EXECUTE_TIMEOUT_CAP_S = 600.0
EXECUTE_MAX_OUTPUT_CHARS = 50_000


@function_tool
async def execute(
    ctx: RunContextWrapper[AgentContext],
    command: str,
    timeout_seconds: float = 120.0,
) -> str:
    """Run a **host shell** command (not the virtual file-tool layer).

    - **Working directory** is the backend’s **project root** (the same tree file tools use:
      paths like ``/test_demo/foo`` in file tools map to ``<root>/test_demo/foo``).
    - Shell commands are passed to the host shell as given; prefer file tools for project files.
    - Requires a backend that implements ``execute`` (e.g. ``LocalShellBackend``). Output is
      capped. Use short, non-interactive commands only.
    - **timeout_seconds** is clamped to a safe upper bound; unsupported backends return a clear
      message instead of running a shell.
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    cmd = (command or "").strip()
    if not cmd:
        return "No command provided."
    t = float(timeout_seconds)
    t = min(max(t, 1.0), EXECUTE_TIMEOUT_CAP_S)
    return b.execute(
        sid,
        cmd,
        timeout=t,
        max_chars=EXECUTE_MAX_OUTPUT_CHARS,
    )
