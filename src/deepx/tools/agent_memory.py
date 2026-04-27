from __future__ import annotations

from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool

from deepx.backends.utils import MAX_READ_FILE_LINES, data_root_as_agent_path
from deepx.context import AgentContext


@function_tool
async def save_memory(ctx: RunContextWrapper[AgentContext], fact: str) -> str:
    """Append a durable fact to agent memory (``AGENTS.md`` under the backend ``data_root``).

    Use this for **cross-conversation** or **cross-session** reminders: preferences, standing
    instructions, project conventions, or anything that should survive new chat threads when the
    same workspace/backend is used. Boot-time ``memory=`` file paths load into the system prompt;
    this tool **appends** bullet lines to ``/.deepx/AGENTS.md`` (the same store the framework
    loads from on later runs).
    """
    text = (fact or "").strip()
    if not text:
        return "No fact provided."
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"\n- [{ts}] {text}\n"
    sid = ctx.context.session_id
    path = data_root_as_agent_path("AGENTS.md")
    try:
        rr = ctx.context.backend.read(sid, path, 0, MAX_READ_FILE_LINES)
        prev = "" if (rr.error or not rr.content) else rr.content
        wr = ctx.context.backend.write(sid, path, prev + line)
        if wr.error:
            return f"Could not save memory: {wr.error}"
    except OSError as e:
        return f"Could not save memory: {e}"
    return "Saved to persistent memory."
