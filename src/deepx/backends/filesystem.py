from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import wcmatch.glob as wcglob
from wcmatch import fnmatch as wc_fnmatch

from deepx.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

# --- Tool-result size helpers (used by middleware eviction) ---

NUM_CHARS_PER_TOKEN = 4
TOOL_RESULT_TOKEN_LIMIT = 20_000
OUTPUTS_LARGE_TOOL_RESULTS_PREFIX = "/_outputs/large_tool_results"


def sanitize_tool_call_id(tool_call_id: str) -> str:
    return tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")


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


def data_root_for_host(host_root: Path) -> Path:
    r = host_root.expanduser().resolve()
    if r.name == ".deepx":
        return r
    return r / ".deepx"


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
    # OS-absolute path that lies under project root → agent path
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


class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir: str | Path) -> None:
        self._host_root = Path(root_dir).expanduser().resolve()
        self._data_root = data_root_for_host(self._host_root)

    @property
    def data_root(self) -> Path:
        return self._data_root

    def _physical(self, agent_path: str) -> tuple[Path | None, str | None]:
        return _physical_for_tools(self._host_root, self._data_root, agent_path)

    def _agent_path_for_file(self, f: Path) -> str | None:
        try:
            rel = f.resolve().relative_to(self._host_root.resolve())
            return "/" + rel.as_posix()
        except ValueError:
            return None

    def ls(self, session_id: str, path: str) -> LsResult:
        _ = session_id
        p, err = self._physical(path)
        if err or p is None:
            return LsResult(error=err or "Error: path not found.")
        if not p.exists():
            return LsResult(error=f"Error: directory '{path}' not found.")
        if not p.is_dir():
            return LsResult(error=f"Error: '{path}' is not a directory.")
        entries: list[FileInfo] = []
        try:
            for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if _under_data_root(child, self._data_root):
                    continue
                ap = self._agent_path_for_file(child)
                if ap is None:
                    continue
                if child.is_dir():
                    entries.append(FileInfo(path=ap, is_dir=True))
                else:
                    st = child.stat()
                    ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    entries.append(
                        FileInfo(path=ap, is_dir=False, size=st.st_size, modified_at=ts)
                    )
        except OSError as e:
            return LsResult(error=f"Error: {e}")
        return LsResult(entries=entries)

    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        _ = session_id
        p, err = self._physical(file_path)
        if err or p is None:
            return ReadResult(error=err or f"Error: '{file_path}' not found.")
        if not p.is_file():
            return ReadResult(error=f"Error: '{file_path}' not found.")
        try:
            raw = p.read_bytes()
        except OSError as e:
            return ReadResult(error=f"Error: {e}")
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        selected = lines[offset : offset + limit]
        if not selected and lines:
            return ReadResult(
                error=f"Error: offset {offset} exceeds file length ({total} lines).",
                total_lines=total,
            )
        return ReadResult(content="\n".join(selected), total_lines=total)

    def grep(
        self,
        session_id: str,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        _ = session_id
        base_agent = _norm_agent_path(path or "/")
        base_phys, err = self._physical(base_agent)
        if err or base_phys is None:
            return GrepResult(error=err or f"Error: path '{base_agent}' not found.")
        if not base_phys.exists():
            return GrepResult(error=f"Error: path '{base_agent}' not found.")
        candidates: list[Path] = []
        _gflags = wcglob.EXTGLOB | wcglob.GLOBSTAR | wcglob.DOTGLOB
        if base_phys.is_file():
            candidates = [base_phys]
        else:
            for f in sorted(base_phys.rglob("*")):
                if not f.is_file():
                    continue
                if _under_data_root(f, self._data_root):
                    continue
                rel = f.relative_to(base_phys).as_posix()
                if glob and not (
                    wc_fnmatch.fnmatch(rel, glob, flags=_gflags)
                    or wc_fnmatch.fnmatch(f.name, glob, flags=_gflags)
                ):
                    continue
                candidates.append(f)
        matches: list[GrepMatch] = []
        for fp in candidates:
            ap = self._agent_path_for_file(fp)
            if ap is None:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=ap, line_number=i, line=line))
        return GrepResult(matches=matches)

    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        _ = session_id
        base_agent = _norm_agent_path(path)
        base, err = self._physical(base_agent)
        if err or base is None:
            return GlobResult(error=err or f"Error: path '{base_agent}' not found.")
        if not base.exists():
            return GlobResult(error=f"Error: path '{base_agent}' not found.")
        files: list[FileInfo] = []
        try:
            _flags = wcglob.GLOBSTAR | wcglob.EXTGLOB | wcglob.BRACE | wcglob.DOTGLOB
            rel_hits = wcglob.glob(
                pattern,
                root_dir=str(base),
                flags=_flags,
                limit=5001,
            )
            for rel in sorted(rel_hits)[:500]:
                f = (base / rel).resolve()
                try:
                    f.relative_to(base.resolve())
                except ValueError:
                    continue
                if _under_data_root(f, self._data_root):
                    continue
                if not f.is_file():
                    continue
                ap = self._agent_path_for_file(f)
                if ap is None:
                    continue
                st = f.stat()
                ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
                files.append(
                    FileInfo(path=ap, is_dir=False, size=st.st_size, modified_at=ts)
                )
        except OSError as e:
            return GlobResult(error=f"Error: {e}")
        files.sort(key=lambda x: x.path)
        return GlobResult(files=files)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        _ = session_id
        canonical = _norm_agent_path(_coerce_agent_path(self._host_root, file_path))
        p, err = self._physical(file_path)
        if err or p is None:
            return WriteResult(error=err or "Cannot write: invalid path.")
        allow_replace = canonical.startswith(OUTPUTS_LARGE_TOOL_RESULTS_PREFIX + "/")
        if p.exists() and not allow_replace:
            return WriteResult(
                error=f"Cannot write to {canonical} because it already exists."
            )
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return WriteResult(error=f"Error: {e}")
        return WriteResult(path=canonical, files_update=None)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        _ = session_id
        p, err = self._physical(file_path)
        if err or p is None:
            return EditResult(error=err or f"Error: '{file_path}' not found.")
        if not p.is_file():
            return EditResult(error=f"Error: '{file_path}' not found.")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return EditResult(error=f"Error: {e}")
        count = content.count(old_string)
        if count == 0:
            return EditResult(error=f"Error: string not found in '{file_path}'.")
        if not replace_all and count > 1:
            return EditResult(
                error=(
                    f"Error: String appears {count} times in file. "
                    "Use replace_all=True or provide a more specific old_string."
                )
            )
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        try:
            p.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return EditResult(error=f"Error: {e}")
        return EditResult(path=file_path, occurrences=count if replace_all else 1)

    def execute(
        self,
        session_id: str,
        command: str,
        *,
        timeout: float = 120.0,
        max_chars: int = 50_000,
    ) -> str:
        _ = session_id, command, timeout, max_chars
        return (
            "Shell execution is not available on FilesystemBackend. "
            "Use LocalShellBackend (or another backend that implements execute) for the execute tool."
        )


def resolve_host_root(backend: BackendProtocol) -> Path | None:
    if isinstance(backend, FilesystemBackend):
        return backend._host_root
    return None


def resolve_data_root(backend: BackendProtocol) -> Path | None:
    if isinstance(backend, FilesystemBackend):
        return backend.data_root
    return None
