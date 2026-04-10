from __future__ import annotations

import json

from deepx.backends.protocol import BackendProtocol


class InMemoryBackend(BackendProtocol):
    def __init__(self) -> None:
        self._files: dict[tuple[str, str], str] = {}
        self._store: dict[str, str] = {}
        self._plans: dict[tuple[str, str], str] = {}
        self._task_logs: dict[str, list[str]] = {}
        self._plan_logs: dict[str, list[dict]] = {}
        self._tool_logs: dict[str, list[dict]] = {}

    def read(self, session_id: str, path: str) -> str | None:
        key = (session_id, path.lstrip("/"))
        return self._files.get(key)

    def write(self, session_id: str, path: str, content: str) -> None:
        key = (session_id, path.lstrip("/"))
        self._files[key] = content

    def append(self, session_id: str, path: str, content: str) -> None:
        key = (session_id, path.lstrip("/"))
        self._files[key] = self._files.get(key, "") + content

    def exists(self, session_id: str, path: str) -> bool:
        key = (session_id, path.lstrip("/"))
        return key in self._files

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        prefix = prefix.lstrip("/")
        keys = [p for sid, p in self._files if sid == session_id]
        return sorted(k for k in keys if not prefix or k.startswith(prefix))

    def read_store(self, path: str) -> str | None:
        return self._store.get(path.lstrip("/"))

    def write_store(self, path: str, content: str) -> None:
        self._store[path.lstrip("/")] = content

    def list_store(self, prefix: str = "") -> list[str]:
        prefix = prefix.lstrip("/")
        return sorted(k for k in self._store if not prefix or k.startswith(prefix))

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        self._plans[(session_id, agent_name)] = plan_json

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        return self._plans.get((session_id, agent_name))

    def append_task_log(self, session_id: str, task: str) -> None:
        self._task_logs.setdefault(session_id, []).append(task)

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        try:
            obj = json.loads(entry_json)
        except json.JSONDecodeError:
            obj = {"raw": entry_json}
        self._plan_logs.setdefault(session_id, []).append(obj)

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        logs = self._tool_logs.setdefault(session_id, [])
        tn = str(log_data["tool_name"])
        n = 1 + sum(1 for e in logs if str(e.get("tool_name")) == tn)
        entry = {**log_data, "call_id": str(n)}
        logs.append(entry)

    def append_system_prompt_log(self, session_id: str, agent_name: str, prompt: str) -> None:
        pass
