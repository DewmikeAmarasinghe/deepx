from __future__ import annotations

import fnmatch
import json
import re
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


def _safe_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "agent"


def _norm_agent_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    if not p.startswith("/"):
        p = "/" + p
    return p


def _split_scope(agent_path: str) -> tuple[str, str]:
    p = _norm_agent_path(agent_path)
    body = p[1:]
    if body.startswith("_workspace_/"):
        return "ws", body[len("_workspace_/") :]
    if body == "_workspace_":
        return "ws", ""
    if body.startswith("store/"):
        return "store", body[6:]
    if body == "store":
        return "store", ""
    return "root", body


class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).expanduser().resolve()

    def _workspace_base(self, session_id: str) -> Path:
        return self._root / ".deepx" / "sessions" / session_id / "files"

    def _logs_dir(self, session_id: str) -> Path:
        return self._root / ".deepx" / "sessions" / session_id / "logs"

    def _memory_base(self) -> Path:
        return self._root / "memory"

    def _physical_dir(self, session_id: str, scope: str, rel: str) -> Path:
        rel = rel.strip("/").replace("\\", "/")
        if scope == "ws":
            return self._workspace_base(session_id) / rel if rel else self._workspace_base(session_id)
        if scope == "store":
            return self._memory_base() / rel if rel else self._memory_base()
        return self._root / rel if rel else self._root

    def _physical_file(self, session_id: str, agent_path: str) -> tuple[str, str, Path]:
        scope, rel = _split_scope(agent_path)
        p = self._physical_dir(session_id, scope, rel)
        return scope, rel, p

    def _file_info(self, agent_prefix: str, physical: Path) -> FileInfo:
        rel = physical.name
        ap = f"{agent_prefix.rstrip('/')}/{rel}" if agent_prefix else f"/{rel}"
        if physical.is_dir():
            return FileInfo(path=ap, is_dir=True)
        st = physical.stat()
        ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        return FileInfo(path=ap, is_dir=False, size=st.st_size, modified_at=ts)

    def ls(self, session_id: str, path: str) -> LsResult:
        p = _norm_agent_path(path)
        scope, rel = _split_scope(p)
        base = self._physical_dir(session_id, scope, rel)
        if not base.exists():
            if scope == "ws":
                return LsResult(entries=[])
            return LsResult(error=f"Error: directory '{path}' not found.")
        if not base.is_dir():
            return LsResult(error=f"Error: '{path}' is not a directory.")
        entries: list[FileInfo] = []
        if p == "/":
            entries.append(FileInfo(path="/_workspace_", is_dir=True))
            entries.append(FileInfo(path="/store", is_dir=True))
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

    def _agent_path_for_file(self, session_id: str, f: Path) -> str | None:
        try:
            rel = f.relative_to(self._workspace_base(session_id))
            return "/_workspace_/" + rel.as_posix()
        except ValueError:
            pass
        try:
            rel = f.relative_to(self._memory_base())
            return "/store/" + rel.as_posix()
        except ValueError:
            pass
        try:
            rel = f.relative_to(self._root)
            return "/" + rel.as_posix()
        except ValueError:
            return None

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
            if _split_scope(base_agent)[0] == "ws":
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
            if scope == "ws":
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
            content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        )
        try:
            p.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return EditResult(error=f"Error: {e}")
        return EditResult(path=file_path, occurrences=count if replace_all else 1)

    def _plan_path(self, session_id: str, agent_name: str) -> Path:
        safe = _safe_agent_name(agent_name)
        return self._logs_dir(session_id) / "plans" / f"{safe}.json"

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        p = self._plan_path(session_id, agent_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(plan_json, encoding="utf-8")

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        p = self._plan_path(session_id, agent_name)
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        p = self._logs_dir(session_id) / "plans" / "plans.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(arr, list):
                    arr = []
            except json.JSONDecodeError:
                arr = []
        else:
            arr = []
        try:
            arr.append(json.loads(entry_json))
        except json.JSONDecodeError:
            arr.append({"raw": entry_json})
        p.write_text(json.dumps(arr, indent=2), encoding="utf-8")

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        tool_name = str(log_data["tool_name"])
        dir_path = self._logs_dir(session_id) / "tools" / tool_name
        dir_path.mkdir(parents=True, exist_ok=True)
        existing = [int(x.stem) for x in dir_path.glob("*.json") if x.stem.isdigit()]
        next_id = max(existing, default=0) + 1
        entry = {**log_data, "call_id": str(next_id)}
        (dir_path / f"{next_id}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
