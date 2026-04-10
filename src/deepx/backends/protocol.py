from __future__ import annotations

import abc


class BackendProtocol(abc.ABC):
    @abc.abstractmethod
    def read(self, session_id: str, path: str) -> str | None: ...

    @abc.abstractmethod
    def write(self, session_id: str, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def append(self, session_id: str, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def exists(self, session_id: str, path: str) -> bool: ...

    @abc.abstractmethod
    def list_files(self, session_id: str, prefix: str = "") -> list[str]: ...

    @abc.abstractmethod
    def read_store(self, path: str) -> str | None: ...

    @abc.abstractmethod
    def write_store(self, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def list_store(self, prefix: str = "") -> list[str]: ...

    @abc.abstractmethod
    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None: ...

    @abc.abstractmethod
    def load_plan(self, session_id: str, agent_name: str) -> str | None: ...

    @abc.abstractmethod
    def append_plan_log(self, session_id: str, entry_json: str) -> None: ...

    @abc.abstractmethod
    def save_tool_log(self, session_id: str, log_data: dict) -> None: ...

    @property
    def supports_execution(self) -> bool:
        return False

    def execute(self, command: str) -> str:
        raise NotImplementedError("Shell execution requires a sandbox backend.")
