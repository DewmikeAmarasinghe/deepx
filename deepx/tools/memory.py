from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def update_memory(ctx: RunContextWrapper[AgentContext], note: str) -> str:
    line = f"- {note}"
    if ctx.context.memory:
        ctx.context.memory = ctx.context.memory.rstrip() + "\n" + line
    else:
        ctx.context.memory = line
    ctx.context.vfs["/memory/notes.md"] = ctx.context.memory
    return "remembered"


@function_tool
async def read_memory(ctx: RunContextWrapper[AgentContext]) -> str:
    return ctx.context.memory if ctx.context.memory else "(empty)"
