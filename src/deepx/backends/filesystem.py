from __future__ import annotations

import json
import re
from pathlib import Path

from deepx.backends.protocol import BackendProtocol


def _safe_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "agent"


class FilesystemBackend(BackendProtocol):
    def __init__(self, root: str | Path = ".deepx") -> None:
        self._root = Path(root)

    def _files_base(self, session_id: str) -> Path:
        return self._root / "sessions" / session_id / "files"

    def _file_path(self, session_id: str, path: str) -> Path:
        rel = path.lstrip("/").replace("\\", "/")
        return self._files_base(session_id) / rel

    def _memory_base(self) -> Path:
        return self._root / "memory"

    def _memory_path(self, path: str) -> Path:
        rel = path.lstrip("/").replace("\\", "/")
        return self._memory_base() / rel

    def _plan_path(self, session_id: str, agent_name: str) -> Path:
        safe = _safe_agent_name(agent_name)
        return self._root / "sessions" / session_id / "plans" / f"{safe}.json"

    def _logs_dir(self, session_id: str) -> Path:
        return self._root / "sessions" / session_id / "logs"

    def _tool_log_path(self, session_id: str, tool_name: str, call_id: str) -> Path:
        return self._logs_dir(session_id) / "tools" / tool_name / f"{call_id}.json"

    def read(self, session_id: str, path: str) -> str | None:
        p = self._file_path(session_id, path)
        return p.read_text() if p.is_file() else None

    def read_session_bytes(self, session_id: str, path: str) -> bytes | None:
        p = self._file_path(session_id, path)
        return p.read_bytes() if p.is_file() else None

    def read_store_bytes(self, path: str) -> bytes | None:
        p = self._memory_path(path)
        return p.read_bytes() if p.is_file() else None

    def write(self, session_id: str, path: str, content: str) -> None:
        p = self._file_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def append(self, session_id: str, path: str, content: str) -> None:
        p = self._file_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)

    def exists(self, session_id: str, path: str) -> bool:
        return self._file_path(session_id, path).is_file()

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._files_base(session_id)
        if not base.exists():
            return []
        prefix = prefix.lstrip("/").replace("\\", "/")
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            if prefix and not rel.startswith(prefix):
                continue
            out.append(rel)
        return out

    def read_store(self, path: str) -> str | None:
        p = self._memory_path(path)
        return p.read_text() if p.is_file() else None

    def write_store(self, path: str, content: str) -> None:
        p = self._memory_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_store(self, prefix: str = "") -> list[str]:
        base = self._memory_base()
        if not base.exists():
            return []
        prefix = prefix.lstrip("/").replace("\\", "/")
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            if prefix and not rel.startswith(prefix):
                continue
            out.append(rel)
        return out

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        p = self._plan_path(session_id, agent_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(plan_json)

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        p = self._plan_path(session_id, agent_name)
        return p.read_text() if p.is_file() else None

    def append_task_log(self, session_id: str, task: str) -> None:
        p = self._logs_dir(session_id) / "tasks.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(task + "\n")

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        p = self._logs_dir(session_id) / "plans.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                arr = json.loads(p.read_text())
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
        p.write_text(json.dumps(arr, indent=2))

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        tool_name = str(log_data["tool_name"])
        call_id = str(log_data["call_id"])
        path = self._tool_log_path(session_id, tool_name, call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log_data, indent=2))

    def append_system_prompt_log(self, session_id: str, agent_name: str, prompt: str) -> None:
        from datetime import datetime, timezone
        p = self._logs_dir(session_id) / "system_prompts.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                arr = json.loads(p.read_text())
                if not isinstance(arr, list):
                    arr = []
            except json.JSONDecodeError:
                arr = []
        else:
            arr = []
        arr.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "prompt": prompt,
        })
        p.write_text(json.dumps(arr, indent=2))
