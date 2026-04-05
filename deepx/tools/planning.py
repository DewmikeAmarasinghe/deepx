from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def write_todos(ctx: RunContextWrapper[AgentContext], items: list[str]) -> str:
    ctx.context.todos = list(items)
    body = "\n".join(f"- {t}" for t in items)
    ctx.context.vfs["/plan.md"] = body
    return f"plan updated ({len(items)} items):\n{body}"


@function_tool
async def mark_done(ctx: RunContextWrapper[AgentContext], index: int) -> str:
    if index < 0 or index >= len(ctx.context.todos):
        return f"error: invalid index {index}"
    item = ctx.context.todos[index]
    if not item.startswith("✓ "):
        ctx.context.todos[index] = f"✓ {item}"
    body = "\n".join(f"- {t}" for t in ctx.context.todos)
    ctx.context.vfs["/plan.md"] = body
    return f"marked done: {ctx.context.todos[index]}"


@function_tool
async def read_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    return ctx.context.vfs.get("/plan.md", "(no plan)")
