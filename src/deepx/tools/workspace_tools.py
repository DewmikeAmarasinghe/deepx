from __future__ import annotations

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
def write_file(ctx: RunContextWrapper[AgentContext], path: str, content: str) -> str:
    """Write content to a new file in the workspace. Path is relative (e.g. 'research/topic.md').
    Returns an error if the file already exists. Use append_to_file to add to an existing file.
    Always prefer writing findings to files rather than holding them in conversation."""
    if ctx.context.backend.exists(ctx.context.session_id, path):
        return f"Error: '{path}' already exists. Use edit_file to modify it."
    ctx.context.backend.write(ctx.context.session_id, path, content)
    return f"Written: {path} ({len(content)} chars)"


@function_tool
def read_file(
    ctx: RunContextWrapper[AgentContext], path: str, offset: int = 0, limit: int = 100
) -> str:
    """Read a file from the workspace with pagination. Path is relative (e.g. 'research/topic.md').
    Use offset and limit to read large files in sections. Returns line-numbered content."""
    content = ctx.context.backend.read(ctx.context.session_id, path)
    if content is None:
        shared = ctx.context.backend.read_shared(path)
        if shared is None:
            return f"Error: '{path}' not found in workspace or shared memory."
        content = shared
    lines = content.splitlines()
    selected = lines[offset : offset + limit]
    if not selected:
        return f"Error: offset {offset} exceeds file length ({len(lines)} lines)."
    numbered = "\n".join(
        f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(selected)
    )
    if len(lines) > offset + limit:
        numbered += f"\n\n[{len(lines) - offset - limit} more lines. Use offset={offset + limit} to continue.]"
    return numbered


@function_tool
def append_to_file(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Append content to an existing file. Creates the file if it does not exist.
    Use this to accumulate research findings, logs, or any growing data."""
    ctx.context.backend.append(ctx.context.session_id, path, "\n" + content)
    return f"Appended to: {path}"


@function_tool
def edit_file(
    ctx: RunContextWrapper[AgentContext], path: str, old_string: str, new_string: str
) -> str:
    """Edit a file by replacing an exact string. The old_string must appear exactly once.
    Always read the file first to confirm the exact text before editing."""
    content = ctx.context.backend.read(ctx.context.session_id, path)
    if content is None:
        return f"Error: '{path}' not found."
    count = content.count(old_string)
    if count == 0:
        return f"Error: string not found in '{path}'."
    if count > 1:
        return f"Error: string appears {count} times. Provide more context to make it unique."
    ctx.context.backend.write(
        ctx.context.session_id, path, content.replace(old_string, new_string, 1)
    )
    return f"Edited: {path}"


@function_tool
def list_files(ctx: RunContextWrapper[AgentContext], prefix: str = "") -> str:
    """List all files in the workspace for this session. Optionally filter by prefix.
    Returns a sorted list of relative file paths. Always call this before reading to discover files."""
    files = ctx.context.backend.list_files(ctx.context.session_id, prefix)
    if not files:
        return "No files found." + (f" (prefix='{prefix}')" if prefix else "")
    return "\n".join(files)
