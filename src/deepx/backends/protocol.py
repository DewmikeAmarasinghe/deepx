from __future__ import annotations
import abc


class WorkspaceBackend(abc.ABC):
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
    def read_shared(self, path: str) -> str | None: ...

    @abc.abstractmethod
    def write_shared(self, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def list_shared(self, prefix: str = "") -> list[str]: ...

    @abc.abstractmethod
    def save_tool_log(self, session_id: str, log_data: dict) -> None: ...

    @abc.abstractmethod
    def save_plan(self, session_id: str, plan_json: str) -> None: ...

    @abc.abstractmethod
    def load_plan(self, session_id: str) -> str | None: ...
