from __future__ import annotations

import base64
import fnmatch
from datetime import datetime, timezone
from typing import Literal

from agents import RunContextWrapper, function_tool

from deepx.backends.filesystem import FilesystemBackend
from deepx.context import AgentContext

_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


def _route_path(path: str) -> tuple[str, str]:
    p = path.strip().replace("\\", "/")
    if p.startswith("/store/"):
        return "store", p[7:].lstrip("/")
    if p.startswith("store/"):
        return "store", p[6:].lstrip("/")
    return "files", p.lstrip("/")


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _path_ext(rel: str) -> str:
    if "." not in rel:
        return ""
    return "." + rel.rsplit(".", 1)[-1].lower()


def _mime(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")


def _child_map(all_rels: list[str], base_prefix: str) -> dict[str, bool]:
    base_prefix = base_prefix.strip("/")
    pfx = f"{base_prefix}/" if base_prefix else ""
    children: dict[str, bool] = {}
    for rel in all_rels:
        if base_prefix:
            if rel == base_prefix:
                continue
            if not rel.startswith(pfx):
                continue
            rest = rel[len(pfx):]
        else:
            rest = rel
        if not rest:
            continue
        head, _, tail = rest.partition("/")
        is_dir = bool(tail) or any(o.startswith(f"{pfx}{head}/") for o in all_rels)
        if head in children:
            children[head] = children[head] or is_dir
        else:
            children[head] = is_dir
    return children


def _format_ls_lines(
    ctx: RunContextWrapper[AgentContext],
    children: dict[str, bool],
    pfx: str,
    *,
    store: bool,
) -> str:
    b = ctx.context.backend
    sid = ctx.context.session_id
    fs_b = b if isinstance(b, FilesystemBackend) else None
    lines: list[str] = []
    for name in sorted(children):
        is_dir = children[name]
        if is_dir:
            lines.append(f"{name}/{' ' * max(0, 22 - len(name))}DIR")
            continue
        rel_path = f"{pfx}{name}" if pfx else name
        ts = ""
        sz = ""
        if store:
            raw = b.read_store(rel_path)
            if raw is not None:
                sz = _format_size(len(raw.encode("utf-8")))
        elif fs_b:
            p_stat = fs_b._file_path(sid, rel_path)
            if p_stat.is_file():
                st = p_stat.stat()
                sz = _format_size(st.st_size)
                ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
        else:
            raw = b.read(sid, rel_path)
            if raw is not None:
                sz = _format_size(len(raw.encode("utf-8")))
        pad = max(0, 18 - len(name))
        lines.append(f"{name}{' ' * pad}{sz}{' ' * 4}{ts}".rstrip())
    return "\n".join(lines) if lines else "(empty)"


def _run_ls(ctx: RunContextWrapper[AgentContext], path: str) -> str:
    kind, rel = _route_path(path)
    base_prefix = rel.strip("/")
    if kind == "store":
        all_rels = ctx.context.backend.list_store("")
        children = _child_map(all_rels, base_prefix)
        pfx = f"{base_prefix}/" if base_prefix else ""
        return _format_ls_lines(ctx, children, pfx, store=True)
    all_rels = ctx.context.backend.list_files(ctx.context.session_id, "")
    children = _child_map(all_rels, base_prefix)
    pfx = f"{base_prefix}/" if base_prefix else ""
    return _format_ls_lines(ctx, children, pfx, store=False)


@function_tool
def ls(ctx: RunContextWrapper[AgentContext], path: str = "/") -> str:
    """Lists all files in a directory.

    This is useful for exploring the filesystem and finding the right file to read or edit.
    You should almost ALWAYS use this tool before using the read_file or edit_file tools."""
    return _run_ls(ctx, path)


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
    - Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) are returned as base64-encoded data.

    - You should ALWAYS make sure a file has been read before editing it."""
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    ext = _path_ext(rel)

    if kind == "store":
        if ext in _IMAGE_EXT and isinstance(b, FilesystemBackend):
            raw = b.read_store_bytes(rel)
            if raw is None:
                return f"Error: '{path}' not found."
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return f"[Image base64 {ext}]\ndata:{_mime(ext)};base64,{b64}"
        content = b.read_store(rel)
    else:
        if ext in _IMAGE_EXT and isinstance(b, FilesystemBackend):
            raw = b.read_session_bytes(sid, rel)
            if raw is None:
                return f"Error: '{path}' not found."
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return f"[Image base64 {ext}]\ndata:{_mime(ext)};base64,{b64}"
        content = b.read(sid, rel)

    if content is None:
        return f"Error: '{path}' not found."

    if not content.strip():
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    selected = lines[offset: offset + limit]
    if not selected and lines:
        return f"Error: offset {offset} exceeds file length ({len(lines)} lines)."
    numbered = "\n".join(
        f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(selected)
    )
    if len(lines) > offset + limit:
        numbered += f"\n\n[{len(lines) - offset - limit} more lines — use offset={offset + limit} to continue]"
    return numbered


@function_tool
def write_file(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Writes to a new file in the filesystem.

    Usage:
    - The write_file tool will create a new file.
    - Prefer to edit existing files (with the edit_file tool) over creating new ones when possible."""
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    if kind == "store":
        if b.read_store(rel) is not None:
            return (
                f"Cannot write to {path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )
        b.write_store(rel, content)
    else:
        if b.exists(sid, rel):
            return (
                f"Cannot write to {path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )
        b.write(sid, rel, content)
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
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    if kind == "store":
        content = b.read_store(rel)
    else:
        content = b.read(sid, rel)
    if content is None:
        return f"Error: '{path}' not found."
    count = content.count(old_string)
    if count == 0:
        return f"Error: string not found in '{path}'."
    if not replace_all and count > 1:
        return (
            f"Error: String '{old_string}' appears {count} times in file. "
            "Use replace_all=True to replace all instances, or provide a more "
            "specific string with surrounding context."
        )
    new_content = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    if kind == "store":
        b.write_store(rel, new_content)
    else:
        b.write(sid, rel, new_content)
    return f"Successfully replaced {count if replace_all else 1} instance(s) of the string in '{path}'"


@function_tool
def glob(
    ctx: RunContextWrapper[AgentContext], pattern: str, path: str = "/"
) -> str:
    """Find files matching a glob pattern.

    Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
    Returns a list of file paths that match the pattern.

    Examples:
    - `**/*.py` - Find all Python files
    - `*.txt` - Find all text files in root
    - `/subdir/**/*.md` - Find all markdown files under /subdir"""
    kind, rel = _route_path(path)
    base_prefix = rel.strip("/")
    pfx = f"{base_prefix}/" if base_prefix else ""
    if kind == "store":
        candidates = ctx.context.backend.list_store("")
        if base_prefix:
            candidates = [
                c
                for c in candidates
                if c == base_prefix or c.startswith(pfx)
            ]
    else:
        sid = ctx.context.session_id
        candidates = ctx.context.backend.list_files(sid, prefix=pfx if pfx else "")
    matches = sorted({p for p in candidates if fnmatch.fnmatch(p, pattern)})
    return "\n".join(matches) if matches else "(no matches)"


@function_tool
def grep(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    path: str | None = None,
    glob_pattern: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """Search for a text pattern across files.

    Searches for literal text (not regex) and returns matching files or content based on output_mode.
    Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

    Examples:
    - Search all files: `grep(pattern="TODO")`
    - Search Python files only: `grep(pattern="import", glob_pattern="*.py")`
    - Show matching lines: `grep(pattern="error", output_mode="content")`
    - Search for code with special chars: `grep(pattern="def __init__(self):")`"""
    sid = ctx.context.session_id
    b = ctx.context.backend

    if path:
        kind, rel = _route_path(path)
        base_prefix = rel.strip("/")
        pfx = f"{base_prefix}/" if base_prefix else ""
        if kind == "store":
            files = b.list_store("")
            if base_prefix:
                files = [
                    f
                    for f in files
                    if f == base_prefix or f.startswith(pfx)
                ]
        else:
            files = b.list_files(sid, prefix=pfx if pfx else "")
    else:
        files = b.list_files(sid, "")
        kind = "files"

    if glob_pattern:
        files = [f for f in files if fnmatch.fnmatch(f, glob_pattern)]

    results: list[str] = []
    for fp in sorted(files):
        if path and _route_path(path)[0] == "store":
            raw = b.read_store(fp)
        else:
            raw = b.read(sid, fp)
        if raw is None:
            continue
        count = raw.count(pattern)
        if count == 0:
            continue
        if output_mode == "count":
            results.append(f"{fp}:{count}")
        elif output_mode == "files_with_matches":
            results.append(fp)
        else:
            for i, line in enumerate(raw.splitlines(), 1):
                if pattern in line:
                    results.append(f"{fp}:{i}:{line}")
    return "\n".join(results) if results else "(no matches)"


@function_tool
def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Executes a shell command in an isolated sandbox environment.

    Usage:
    Executes a given command in the sandbox environment with proper handling and security measures.
    Before executing the command, please follow these steps:
    1. Directory Verification:
       - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
    2. Command Execution:
       - Always quote file paths that contain spaces with double quotes
       - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines
    Usage notes:
      - Commands run in an isolated sandbox environment
      - Returns combined stdout/stderr output with exit code
      - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.

    Note: This tool is only available if the backend supports execution."""
    b = ctx.context.backend
    if not b.supports_execution:
        return (
            "Shell execution is not available. Use a sandbox backend to enable it."
        )
    return b.execute(command)
