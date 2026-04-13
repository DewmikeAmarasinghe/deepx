from __future__ import annotations

from deepx.backends.protocol import (
    BackendProtocol,
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)


class CompositeBackend(BackendProtocol):
    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol],
    ) -> None:
        self._default = default
        self._routes = sorted(
            ((p, b) for p, b in routes.items() if p),
            key=lambda x: -len(x[0]),
        )

    def _pick(self, agent_path: str) -> tuple[BackendProtocol, str]:
        p = agent_path if agent_path.startswith("/") else "/" + agent_path
        for prefix, backend in self._routes:
            if p.startswith(prefix):
                rest = p[len(prefix) :].lstrip("/")
                return backend, "/" + rest if rest else "/"
        return self._default, p

    def ls(self, session_id: str, path: str) -> LsResult:
        b, p = self._pick(path)
        return b.ls(session_id, p)

    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        b, p = self._pick(file_path)
        return b.read(session_id, p, offset, limit)

    def grep(
        self,
        session_id: str,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        if path is None:
            return self._default.grep(session_id, pattern, None, glob)
        b, p = self._pick(path)
        return b.grep(session_id, pattern, p, glob)

    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        b, p = self._pick(path)
        return b.glob(session_id, pattern, p)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        b, p = self._pick(file_path)
        return b.write(session_id, p, content)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        b, p = self._pick(file_path)
        return b.edit(session_id, p, old_string, new_string, replace_all)

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        self._default.save_plan(session_id, agent_name, plan_json)

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        return self._default.load_plan(session_id, agent_name)

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        self._default.append_plan_log(session_id, entry_json)

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        self._default.save_tool_log(session_id, log_data)

    @property
    def supports_execution(self) -> bool:
        return self._default.supports_execution

    def execute(self, command: str) -> str:
        return self._default.execute(command)
