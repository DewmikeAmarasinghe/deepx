from __future__ import annotations

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
def update_memory(ctx: RunContextWrapper[AgentContext], note: str) -> str:
    """Append a durable note to agent memory and AGENTS.md in the store."""
    ctx.context.memory = (ctx.context.memory or "") + f"\n- {note}"
    ctx.context.backend.write_store("AGENTS.md", ctx.context.memory)
    return f"Memory updated: {note[:100]}"


@function_tool
def read_memory(ctx: RunContextWrapper[AgentContext]) -> str:
    """Return loaded agent memory text."""
    return ctx.context.memory or "No memory notes yet."


@function_tool
def read_store(ctx: RunContextWrapper[AgentContext], path: str) -> str:
    """Read a file from the cross-session memory store."""
    rel = path.lstrip("/")
    raw = ctx.context.backend.read_store(rel)
    if raw is None:
        return f"Error: '{path}' not found in store."
    return raw


@function_tool
def write_store(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Write content to the cross-session memory store."""
    rel = path.lstrip("/")
    ctx.context.backend.write_store(rel, content)
    return f"Written store: {path} ({len(content)} chars)"
