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
            rest = rel[len(pfx) :]
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
    """List files and directories for the session filesystem or /store/ memory."""
    return _run_ls(ctx, path)


@function_tool
def read_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Read a file from the session or /store/ with line offset and limit."""
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

    lines = content.splitlines()
    selected = lines[offset : offset + limit]
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
    """Create a new file. Fails if the file already exists."""
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
    return f"Written: {path} ({len(content)} chars)"


@function_tool
def edit_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace old_string with new_string in a file (session or /store/)."""
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
    return f"Edited: {path}"


@function_tool
def append_to_file(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Append to a file, creating it if missing."""
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    if kind == "store":
        prev = b.read_store(rel) or ""
        b.write_store(rel, prev + content)
    else:
        b.append(sid, rel, content)
    return f"Appended to: {path}"


@function_tool
def glob(
    ctx: RunContextWrapper[AgentContext], pattern: str, path: str = "/"
) -> str:
    """Find files matching a glob pattern under path (session or /store/)."""
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
    """Literal substring search (not regex). Session files or /store/ path."""
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
def list_files(ctx: RunContextWrapper[AgentContext], path: str = "/") -> str:
    """Deprecated alias for ls."""
    return _run_ls(ctx, path)


@function_tool
def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Run a shell command when execution is enabled on the backend."""
    b = ctx.context.backend
    if not b.supports_execution:
        return (
            "Shell execution is not available. Use a sandbox backend to enable it."
        )
    return b.execute(command)
