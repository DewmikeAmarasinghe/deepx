from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def write_file(ctx: RunContextWrapper[AgentContext], path: str, content: str) -> str:
    if path in ctx.context.vfs:
        return f"error: path exists: {path}"
    ctx.context.vfs[path] = content
    return f"written: {path}"


@function_tool
async def read_file(
    ctx: RunContextWrapper[AgentContext], path: str, offset: int = 0, limit: int = 100
) -> str:
    if path not in ctx.context.vfs:
        return f"error: not found: {path}"
    lines = ctx.context.vfs[path].splitlines()
    chunk = lines[offset : offset + limit]
    out = []
    for i, line in enumerate(chunk, start=offset + 1):
        out.append(f"  {i}\t{line}")
    return "\n".join(out) if out else "(empty range)"


@function_tool
async def edit_file(
    ctx: RunContextWrapper[AgentContext], path: str, old_string: str, new_string: str
) -> str:
    if path not in ctx.context.vfs:
        return f"error: not found: {path}"
    content = ctx.context.vfs[path]
    if old_string not in content:
        return "error: old_string not found in file"
    ctx.context.vfs[path] = content.replace(old_string, new_string, 1)
    return f"edited: {path}"


@function_tool
async def ls(ctx: RunContextWrapper[AgentContext], path: str = "/") -> str:
    keys = sorted(p for p in ctx.context.vfs if p.startswith(path))
    return "\n".join(keys) if keys else "(empty)"


@function_tool
async def append_to_file(ctx: RunContextWrapper[AgentContext], path: str, content: str) -> str:
    if path in ctx.context.vfs:
        ctx.context.vfs[path] = ctx.context.vfs[path] + "\n" + content
    else:
        ctx.context.vfs[path] = content
    return f"appended to: {path}"
