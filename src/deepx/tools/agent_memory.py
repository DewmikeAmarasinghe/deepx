from __future__ import annotations

from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool

from deepx.backends.filesystem import resolve_data_root
from deepx.context import AgentContext


@function_tool
async def save_memory(ctx: RunContextWrapper[AgentContext], fact: str) -> str:
    """Append a fact to persistent agent memory. Loaded automatically on new runs; not under project file paths."""
    text = (fact or "").strip()
    if not text:
        return "No fact provided."
    dr = resolve_data_root(ctx.context.backend)
    if dr is None:
        return "Persistent memory is not available for this backend."
    p = dr / "AGENTS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"\n- [{ts}] {text}\n"
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        return f"Could not save memory: {e}"
    return "Saved to persistent memory."
