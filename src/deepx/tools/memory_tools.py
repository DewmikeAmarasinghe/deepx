from __future__ import annotations
from agents import function_tool, RunContextWrapper
from deepx.context import AgentContext


@function_tool
def update_memory(ctx: RunContextWrapper[AgentContext], note: str) -> str:
    """Add a persistent note to shared memory. This memory is loaded at the start of every session
    and is shared across all agents. Use for important facts, preferences, credentials, patterns,
    or anything that should persist across sessions. Do not store secrets or API keys."""
    ctx.context.memory = (ctx.context.memory or "") + f"\n- {note}"
    ctx.context.backend.write_shared("AGENTS.md", ctx.context.memory)
    return f"Memory updated: {note[:100]}"


@function_tool
def read_memory(ctx: RunContextWrapper[AgentContext]) -> str:
    """Read all notes from shared memory."""
    return ctx.context.memory or "No memory notes yet."
