from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Literal

from agents import RunContextWrapper, function_tool

from deepx.backends.protocol import GrepMatch
from deepx.context import AgentContext

# --- Formatting / limits for file-tool responses ---

GREP_DISPLAY_MAX_MATCHES = 500
GREP_OUTPUT_MAX_CHARS = 100_000
GLOB_TIMEOUT_S = 120.0
READ_MAX_LINE_CHARS = 4000
READ_RESPONSE_MAX_CHARS = 250_000


def _truncate_line(line: str, max_chars: int = READ_MAX_LINE_CHARS) -> str:
    if len(line) <= max_chars:
        return line
    return line[:max_chars] + "… [line truncated]"


def _cap_message(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n… [output truncated]"


def _format_numbered_lines(
    lines: list[str],
    *,
    offset: int,
    max_line_chars: int = READ_MAX_LINE_CHARS,
) -> str:
    parts: list[str] = []
    for i, line in enumerate(lines):
        parts.append(f"{offset + i + 1:6d}\t{_truncate_line(line, max_line_chars)}")
    return "\n".join(parts)


def _grep_matches_as_content(matches: list[GrepMatch], *, max_lines: int) -> str:
    lines_out: list[str] = []
    for m in matches[:max_lines]:
        lines_out.append(
            f"{m.path}:{m.line_number}:{_truncate_line(m.line, READ_MAX_LINE_CHARS)}"
        )
    if len(matches) > max_lines:
        lines_out.append(f"... and {len(matches) - max_lines} more matches.")
    body = "\n".join(lines_out)
    return _cap_message(body, GREP_OUTPUT_MAX_CHARS)


def _grep_matches_as_files(matches: Iterable[GrepMatch]) -> str:
    seen: list[str] = []
    found: set[str] = set()
    for m in matches:
        if m.path not in found:
            found.add(m.path)
            seen.append(m.path)
    body = "\n".join(seen[:2000])
    if len(seen) > 2000:
        body += f"\n... and {len(seen) - 2000} more paths."
    return _cap_message(body, GREP_OUTPUT_MAX_CHARS)


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
    max_line_chars: int = READ_MAX_LINE_CHARS,
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
    - Results are returned using cat -n format, with line numbers starting at 1.
    - Very long lines are truncated with a clear suffix; the full response is capped to avoid
      context blowups.
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
    mlc = max(200, min(int(max_line_chars), 50_000))
    lines = content.splitlines()
    total = rr.total_lines if rr.total_lines is not None else offset + len(lines)
    numbered = _format_numbered_lines(lines, offset=offset, max_line_chars=mlc)
    if total > offset + len(lines):
        numbered += f"\n\n[{total - offset - len(lines)} more lines — use offset={offset + len(lines)} to continue]"
    return _cap_message(numbered, READ_RESPONSE_MAX_CHARS)


@function_tool
def write_file(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Writes to a new file in the filesystem.

    Usage:
    - The write_file tool will create a new file.
    - Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
    - Replacing an existing path is only allowed under ``/_outputs/large_tool_results/`` (large
      evicted tool payloads)."""
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
    output_mode: Literal["content", "count", "files_with_matches"] = "content",
) -> str:
    """Search file contents for a **literal substring** (not a regex).

    - **path:** directory or file to search (agent path, default session workspace root).
    - **glob_pattern:** optional filter so only paths matching this glob are scanned (supports
      ``*``, ``**``, brace expansion—same family as ``glob``).
    - **output_mode:** ``content`` (default) returns ``path:line:`` hits; ``count`` returns match
      counts; ``files_with_matches`` lists paths that contain at least one hit.
    - Skips binary/unreadable files; text is read as UTF-8 with replacement for invalid bytes.
    """
    gr = ctx.context.backend.grep(
        ctx.context.session_id, pattern, path=path, glob=glob_pattern
    )
    if gr.error:
        return gr.error
    if not gr.matches:
        return "No matches."
    if output_mode == "count":
        return f"{len(gr.matches)} matches."
    if output_mode == "files_with_matches":
        return _grep_matches_as_files(gr.matches)
    return _grep_matches_as_content(gr.matches, max_lines=GREP_DISPLAY_MAX_MATCHES)


@function_tool
async def glob(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    path: str = "/",
) -> str:
    """List files under ``path`` whose names match ``pattern`` (glob / wcmatch semantics).

    - **path:** directory under the project root (e.g. ``/``, ``/src``).
    - **pattern:** glob relative to that directory (e.g. ``**/*.py``, ``*.md``). Extended glob
      features (``**``, braces, etc.) are supported.
    - Returns up to 500 paths, one per line, sorted; says if more matched.
    - Aborts with a clear error if the scan exceeds an internal time limit.
    """
    sid = ctx.context.session_id
    try:
        gr = await asyncio.wait_for(
            asyncio.to_thread(ctx.context.backend.glob, sid, pattern, path),
            timeout=GLOB_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return f"Error: glob timed out after {GLOB_TIMEOUT_S:.0f}s (narrow the pattern or path)."
    if gr.error:
        return gr.error
    if not gr.files:
        return "No files matched."
    return "\n".join(f.path for f in gr.files[:500]) + (
        f"\n... and {len(gr.files) - 500} more." if len(gr.files) > 500 else ""
    )
