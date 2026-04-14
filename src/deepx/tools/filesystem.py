from __future__ import annotations

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


@function_tool
def ls(ctx: RunContextWrapper[AgentContext], path: str = "/") -> str:
    """List files in a directory. Use before read_file or edit_file to explore the filesystem."""
    r = ctx.context.backend.ls(ctx.context.session_id, path)
    if r.error:
        return r.error
    if not r.entries:
        return "(empty)"
    lines: list[str] = []
    for e in r.entries:
        name = e.path.rstrip("/").rsplit("/", 1)[-1]
        if e.is_dir:
            pad = max(0, 22 - len(name))
            lines.append(f"{name}/{' ' * pad}DIR")
        else:
            sz = _format_size(e.size or 0)
            ts = e.modified_at or ""
            pad = max(0, 18 - len(name))
            lines.append(f"{name}{' ' * pad}{sz}{' ' * 4}{ts}".rstrip())
    return "\n".join(lines)


@function_tool
def read_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Reads a file from the filesystem.

    Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

    Usage:
    - By default, it reads up to 100 lines starting from the beginning of the file
    - **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
      - First scan: read_file(path, limit=100) to see file structure
      - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
      - Only omit limit (read full file) when necessary for editing
    - Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
    - Results are returned using cat -n format, with line numbers starting at 1
    - You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
    - If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
    - Binary files are decoded as UTF-8 with replacement characters where needed.

    - You should ALWAYS make sure a file has been read before editing it."""
    b = ctx.context.backend
    sid = ctx.context.session_id
    rr = b.read(sid, path, offset=offset, limit=limit)
    if rr.error:
        return rr.error
    content = rr.content or ""
    if not content.strip():
        return "File exists but has empty contents."
    lines = content.splitlines()
    total = rr.total_lines if rr.total_lines is not None else offset + len(lines)
    numbered = "\n".join(
        f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(lines)
    )
    if total > offset + len(lines):
        numbered += f"\n\n[{total - offset - len(lines)} more lines — use offset={offset + len(lines)} to continue]"
    return numbered


@function_tool
def write_file(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Writes to a new file in the filesystem.

    Usage:
    - The write_file tool will create a new file.
    - Prefer to edit existing files (with the edit_file tool) over creating new ones when possible."""
    wr = ctx.context.backend.write(ctx.context.session_id, path, content)
    if wr.error:
        return wr.error
    return f"Updated file {path}"


@function_tool
def edit_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Performs exact string replacements in files.

    Usage:
    - You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
    - When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
    - ALWAYS prefer editing existing files over creating new ones.
    - Only use emojis if the user explicitly requests it."""
    er = ctx.context.backend.edit(
        ctx.context.session_id, path, old_string, new_string, replace_all=replace_all
    )
    if er.error:
        return er.error
    return f"Replaced {er.occurrences} instance(s) in '{path}'"


@function_tool
def grep(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    path: str | None = None,
    glob_pattern: str | None = None,
) -> str:
    """Search file contents for a **literal substring** (not a regex).

    - **path:** directory or file to search (agent path, default session workspace root).
    - **glob_pattern:** optional filter so only paths matching this glob are scanned (supports
      ``*``, ``**``, brace expansion—same family as ``glob``).
    - Returns ``path:line_number:line`` per hit, up to 500 matches (then a truncation notice).
    - Skips binary/unreadable files; text is read as UTF-8 with replacement for invalid bytes.
    """
    gr = ctx.context.backend.grep(
        ctx.context.session_id, pattern, path=path, glob=glob_pattern
    )
    if gr.error:
        return gr.error
    if not gr.matches:
        return "No matches."
    lines_out: list[str] = []
    for m in gr.matches[:500]:
        lines_out.append(f"{m.path}:{m.line_number}:{m.line}")
    if len(gr.matches) > 500:
        lines_out.append(f"... and {len(gr.matches) - 500} more matches.")
    return "\n".join(lines_out)


@function_tool
def glob(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    path: str = "/",
) -> str:
    """List files under ``path`` whose names match ``pattern`` (glob / wcmatch semantics).

    - **path:** directory under the project root (e.g. ``/``, ``/src``).
    - **pattern:** glob relative to that directory (e.g. ``**/*.py``, ``*.md``). Extended glob
      features (``**``, braces, etc.) are supported.
    - Returns up to 500 paths, one per line, sorted; says if more matched.
    """
    gr = ctx.context.backend.glob(ctx.context.session_id, pattern, path)
    if gr.error:
        return gr.error
    if not gr.files:
        return "No files matched."
    return "\n".join(f.path for f in gr.files[:500]) + (
        f"\n... and {len(gr.files) - 500} more." if len(gr.files) > 500 else ""
    )
