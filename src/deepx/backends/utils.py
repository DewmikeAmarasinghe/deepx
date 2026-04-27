"""Shared backend helpers: eviction constants, data-root paths, resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --- Tool-result size / eviction (middleware + backends) ---

NUM_CHARS_PER_TOKEN = 4
TOOL_RESULT_TOKEN_LIMIT = 20_000

OUTPUTS_PREFIX = "/_outputs"
OUTPUTS_LARGE_TOOL_RESULTS_PREFIX = "/_outputs/large_tool_results"

TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)

TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""


def create_large_tool_result_preview(
    content_str: str, *, head_lines: int = 5, tail_lines: int = 5
) -> str:
    lines = content_str.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return "\n".join(line[:1000] for line in lines)

    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]
    omitted = len(lines) - head_lines - tail_lines
    return (
        "\n".join(head)
        + f"\n\n... [{omitted} lines truncated] ...\n\n"
        + "\n".join(tail)
    )


def tool_result_char_budget(*, token_limit: int | None = None) -> int | None:
    if token_limit is None:
        return None
    return NUM_CHARS_PER_TOKEN * int(token_limit)


# Whole-file reads (``read`` tool line limit) for metadata and small JSON blobs.
MAX_READ_FILE_LINES = 10_000_000


def data_root_for_host(root_dir: Path) -> Path:
    """Return ``<root_dir>/.deepx`` (or ``root_dir`` if it is already named ``.deepx``)."""
    r = root_dir.expanduser().resolve()
    if r.name == ".deepx":
        return r
    return r / ".deepx"


def data_root_as_agent_path(rel_under_data_root: str) -> str:
    """Map a path relative to ``data_root`` to the agent path used by ``read`` / ``write`` / ``glob``.

    Example: ``sessions/id/logs/plans/x.json`` → ``/.deepx/sessions/id/logs/plans/x.json``.
    """
    inner = (rel_under_data_root or "").replace("\\", "/").strip().strip("/")
    return f"/.deepx/{inner}" if inner else "/.deepx"


def resolve_root_dir(backend: Any) -> Path | None:
    """Workspace / project root for prompts and memory path resolution."""
    rd = getattr(backend, "_root_dir", None)
    if isinstance(rd, Path):
        return rd
    return None


def resolve_data_root(backend: Any) -> Path | None:
    """Logical ``.deepx`` tree (on-disk path or virtual root for in-memory backends)."""
    dr = getattr(backend, "data_root", None)
    if isinstance(dr, Path):
        return dr
    return None


def resolve_backend_paths(backend: Any) -> tuple[Path | None, Path | None]:
    """Return ``(workspace root, data_root)`` for a backend, if exposed."""
    return resolve_root_dir(backend), resolve_data_root(backend)


# --- Agent path normalization (filesystem + memory backends) ---


def _norm_agent_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    if not p.startswith("/"):
        p = "/" + p
    return p


def _rel_from_agent_path(agent_path: str) -> str:
    return _norm_agent_path(agent_path).lstrip("/")


def _coerce_agent_path(host_root: Path, raw: str) -> str:
    """Normalize user/model paths: virtual ``/…`` under host, or absolute host paths → ``/rel``."""
    s = (raw or "").replace("\\", "/").strip()
    if not s:
        return "/"
    hr = host_root.expanduser().resolve()
    if s.startswith("/") and len(s) > 1:
        try:
            candidate = Path(s).expanduser().resolve()
            rel = candidate.relative_to(hr)
            return "/" + rel.as_posix()
        except (ValueError, OSError):
            pass
    return _norm_agent_path(s)


def _under_data_root(physical: Path, data_root: Path) -> bool:
    try:
        pr = physical.resolve()
        dr = data_root.resolve()
    except OSError:
        return True
    return pr == dr or pr.is_relative_to(dr)


def _physical_for_tools(
    host_root: Path, data_root: Path, agent_path: str
) -> tuple[Path | None, str | None]:
    rel = _rel_from_agent_path(
        _norm_agent_path(_coerce_agent_path(host_root, agent_path))
    )
    hr = host_root.expanduser().resolve()
    try:
        p = (hr / rel if rel else hr).resolve()
    except OSError:
        return None, "Error: invalid path."
    try:
        p.relative_to(hr)
    except ValueError:
        return None, "Error: path escapes project root."
    if _under_data_root(p, data_root):
        return None, "Error: paths under .deepx are not accessible via file tools."
    return p, None


__all__ = [
    "MAX_READ_FILE_LINES",
    "data_root_as_agent_path",
    "NUM_CHARS_PER_TOKEN",
    "OUTPUTS_LARGE_TOOL_RESULTS_PREFIX",
    "OUTPUTS_PREFIX",
    "TOOLS_EXCLUDED_FROM_EVICTION",
    "TOOL_RESULT_TOKEN_LIMIT",
    "TOO_LARGE_TOOL_MSG",
    "create_large_tool_result_preview",
    "data_root_for_host",
    "resolve_backend_paths",
    "resolve_data_root",
    "resolve_root_dir",
    "tool_result_char_budget",
    "_coerce_agent_path",
    "_norm_agent_path",
    "_physical_for_tools",
    "_rel_from_agent_path",
    "_under_data_root",
]
