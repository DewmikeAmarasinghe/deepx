from __future__ import annotations
import json
from pathlib import Path
from deepx.backends.protocol import WorkspaceBackend


class FilesystemBackend(WorkspaceBackend):
    def __init__(self, root: str | Path = ".deepx") -> None:
        self._root = Path(root)

    def _files_path(self, session_id: str, path: str) -> Path:
        return self._root / "sessions" / session_id / "files" / path

    def _tools_path(self, session_id: str, tool_name: str, call_id: str) -> Path:
        return self._root / "sessions" / session_id / "tools" / tool_name / f"{call_id}.json"

    def _plan_path(self, session_id: str) -> Path:
        return self._root / "sessions" / session_id / "plan.json"

    def _shared_path(self, path: str) -> Path:
        return self._root / "memory" / path

    def read(self, session_id: str, path: str) -> str | None:
        p = self._files_path(session_id, path)
        return p.read_text() if p.exists() else None

    def write(self, session_id: str, path: str, content: str) -> None:
        p = self._files_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def append(self, session_id: str, path: str, content: str) -> None:
        p = self._files_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)

    def exists(self, session_id: str, path: str) -> bool:
        return self._files_path(session_id, path).exists()

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._root / "sessions" / session_id / "files"
        if not base.exists():
            return []
        results = [
            str(p.relative_to(base))
            for p in sorted(base.rglob("*"))
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        ]
        return results

    def read_shared(self, path: str) -> str | None:
        p = self._shared_path(path)
        return p.read_text() if p.exists() else None

    def write_shared(self, path: str, content: str) -> None:
        p = self._shared_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_shared(self, prefix: str = "") -> list[str]:
        base = self._root / "memory"
        if not base.exists():
            return []
        return [
            str(p.relative_to(base))
            for p in sorted(base.rglob("*"))
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        ]

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        tool_name = log_data["tool_name"]
        call_id = log_data["call_id"]
        p = self._tools_path(session_id, tool_name, call_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(log_data, indent=2))

    def save_plan(self, session_id: str, plan_json: str) -> None:
        p = self._plan_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(plan_json)

    def load_plan(self, session_id: str) -> str | None:
        p = self._plan_path(session_id)
        return p.read_text() if p.exists() else None
