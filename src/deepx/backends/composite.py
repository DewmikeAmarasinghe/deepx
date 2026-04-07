from __future__ import annotations

from deepx.backends.protocol import WorkspaceBackend


class CompositeBackend(WorkspaceBackend):
    def __init__(
        self,
        default: WorkspaceBackend,
        routes: dict[str, WorkspaceBackend],
    ) -> None:
        self._default = default
        self._routes = sorted(
            ((p, b) for p, b in routes.items() if p),
            key=lambda x: -len(x[0]),
        )

    def _pick(self, path: str) -> tuple[WorkspaceBackend, str]:
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                return backend, path[len(prefix) :].lstrip("/")
        return self._default, path

    def read(self, session_id: str, path: str) -> str | None:
        b, p = self._pick(path)
        return b.read(session_id, p)

    def write(self, session_id: str, path: str, content: str) -> None:
        b, p = self._pick(path)
        b.write(session_id, p, content)

    def append(self, session_id: str, path: str, content: str) -> None:
        b, p = self._pick(path)
        b.append(session_id, p, content)

    def exists(self, session_id: str, path: str) -> bool:
        b, p = self._pick(path)
        return b.exists(session_id, p)

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        b, p = self._pick(prefix or "")
        return b.list_files(session_id, p)

    def read_store(self, path: str) -> str | None:
        return self._default.read_store(path)

    def write_store(self, path: str, content: str) -> None:
        self._default.write_store(path, content)

    def list_store(self, prefix: str = "") -> list[str]:
        return self._default.list_store(prefix)

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        self._default.save_plan(session_id, agent_name, plan_json)

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        return self._default.load_plan(session_id, agent_name)

    def append_task_log(self, session_id: str, task: str) -> None:
        self._default.append_task_log(session_id, task)

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        self._default.append_plan_log(session_id, entry_json)

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        self._default.save_tool_log(session_id, log_data)

    @property
    def supports_execution(self) -> bool:
        return self._default.supports_execution

    def execute(self, command: str) -> str:
        return self._default.execute(command)
