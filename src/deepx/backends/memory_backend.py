from __future__ import annotations
from deepx.backends.protocol import WorkspaceBackend


class InMemoryBackend(WorkspaceBackend):
    def __init__(self) -> None:
        self._files: dict[str, dict[str, str]] = {}
        self._session: dict[str, dict[str, str]] = {}
        self._shared: dict[str, str] = {}
        self._tools: dict[str, list] = {}
        self._plans: dict[str, str] = {}

    def read(self, session_id: str, path: str) -> str | None:
        if path.startswith("../"):
            return self._session.get(session_id, {}).get(path[3:])
        return self._files.get(session_id, {}).get(path)

    def write(self, session_id: str, path: str, content: str) -> None:
        if path.startswith("../"):
            self._session.setdefault(session_id, {})[path[3:]] = content
        else:
            self._files.setdefault(session_id, {})[path] = content

    def append(self, session_id: str, path: str, content: str) -> None:
        existing = self._files.get(session_id, {}).get(path, "")
        self._files.setdefault(session_id, {})[path] = existing + content

    def exists(self, session_id: str, path: str) -> bool:
        return path in self._files.get(session_id, {})

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        return sorted(k for k in self._files.get(session_id, {}) if k.startswith(prefix))

    def read_shared(self, path: str) -> str | None:
        return self._shared.get(path)

    def write_shared(self, path: str, content: str) -> None:
        self._shared[path] = content

    def list_shared(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._shared if k.startswith(prefix))

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        self._tools.setdefault(session_id, []).append(log_data)

    def save_plan(self, session_id: str, plan_json: str) -> None:
        self._plans[session_id] = plan_json

    def load_plan(self, session_id: str) -> str | None:
        return self._plans.get(session_id)