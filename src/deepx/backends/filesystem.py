from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from pathlib import Path

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
from deepx.data_layout import data_root_for_host


def _norm_agent_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    if not p.startswith("/"):
        p = "/" + p
    return p


def _split_scope(agent_path: str) -> tuple[str, str]:
    p = _norm_agent_path(agent_path)
    body = p[1:]
    if body.startswith("_workspace_/"):
        return "_workspace_", body[len("_workspace_/") :]
    if body == "_workspace_":
        return "_workspace_", ""
    if body.startswith("_memory_/"):
        return "_memory_", body[len("_memory_/") :]
    if body == "_memory_":
        return "_memory_", ""
    return "root", body


class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir: str | Path) -> None:
        self._host_root = Path(root_dir).expanduser().resolve()
        self._data_root = data_root_for_host(self._host_root)

    @property
    def data_root(self) -> Path:
        return self._data_root

    def workspace_dir(self, session_id: str) -> Path:
        return self._data_root / "sessions" / session_id / "_workspace_"

    def _memory_dir(self) -> Path:
        return self._data_root / "memory"

    def _physical_dir(self, session_id: str, scope: str, rel: str) -> Path:
        rel = rel.strip("/").replace("\\", "/")
        if scope == "_workspace_":
            base = self.workspace_dir(session_id)
            return base / rel if rel else base
        if scope == "_memory_":
            base = self._memory_dir()
            return base / rel if rel else base
        return self._host_root / rel if rel else self._host_root

    def _physical_file(self, session_id: str, agent_path: str) -> tuple[str, str, Path]:
        scope, rel = _split_scope(agent_path)
        p = self._physical_dir(session_id, scope, rel)
        return scope, rel, p

    def _agent_path_for_file(self, session_id: str, f: Path) -> str | None:
        try:
            rel = f.relative_to(self.workspace_dir(session_id))
            return "/_workspace_/" + rel.as_posix()
        except ValueError:
            pass
        try:
            rel = f.relative_to(self._memory_dir())
            return "/_memory_/" + rel.as_posix()
        except ValueError:
            pass
        try:
            rel = f.relative_to(self._host_root)
            return "/" + rel.as_posix()
        except ValueError:
            return None

    def ls(self, session_id: str, path: str) -> LsResult:
        p = _norm_agent_path(path)
        scope, rel = _split_scope(p)
        base = self._physical_dir(session_id, scope, rel)
        if not base.exists():
            if scope == "_workspace_":
                return LsResult(entries=[])
            return LsResult(error=f"Error: directory '{path}' not found.")
        if not base.is_dir():
            return LsResult(error=f"Error: '{path}' is not a directory.")
        entries: list[FileInfo] = []
        if p == "/":
            entries.append(FileInfo(path="/_workspace_", is_dir=True))
            entries.append(FileInfo(path="/_memory_", is_dir=True))
            try:
                for child in sorted(self._host_root.iterdir(), key=lambda x: x.name.lower()):
                    ap = "/" + child.name
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
        try:
            for child in sorted(base.iterdir(), key=lambda x: x.name.lower()):
                ap = f"{p.rstrip('/')}/{child.name}"
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
        _, _, p = self._physical_file(session_id, file_path)
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
        base_agent = _norm_agent_path(path or "/_workspace_/")
        _, _, base_phys = self._physical_file(session_id, base_agent)
        if not base_phys.exists():
            if _split_scope(base_agent)[0] == "_workspace_":
                return GrepResult(matches=[])
            return GrepResult(error=f"Error: path '{base_agent}' not found.")
        candidates: list[Path] = []
        if base_phys.is_file():
            candidates = [base_phys]
        else:
            for f in sorted(base_phys.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(base_phys).as_posix()
                if glob and not (
                    fnmatch.fnmatch(f.name, glob)
                    or fnmatch.fnmatch(rel, glob)
                    or fnmatch.fnmatch(rel.split("/")[-1], glob)
                ):
                    continue
                candidates.append(f)
        matches: list[GrepMatch] = []
        for fp in candidates:
            ap = self._agent_path_for_file(session_id, fp)
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
        base_agent = _norm_agent_path(path)
        scope, rel = _split_scope(base_agent)
        base = self._physical_dir(session_id, scope, rel)
        if not base.exists():
            if scope == "_workspace_":
                return GlobResult(files=[])
            return GlobResult(error=f"Error: path '{base_agent}' not found.")
        files: list[FileInfo] = []
        try:
            for f in base.glob(pattern):
                if not f.is_file():
                    continue
                ap = self._agent_path_for_file(session_id, f)
                if ap is None:
                    continue
                st = f.stat()
                ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                files.append(FileInfo(path=ap, is_dir=False, size=st.st_size, modified_at=ts))
        except OSError as e:
            return GlobResult(error=f"Error: {e}")
        files.sort(key=lambda x: x.path)
        return GlobResult(files=files)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        _, _, p = self._physical_file(session_id, file_path)
        if p.exists():
            return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as e:
            return WriteResult(error=f"Error: {e}")
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        _, _, p = self._physical_file(session_id, file_path)
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
