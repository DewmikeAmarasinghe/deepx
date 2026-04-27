from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from deepx.backends.utils import (
    NUM_CHARS_PER_TOKEN,
    OUTPUTS_LARGE_TOOL_RESULTS_PREFIX,
    OUTPUTS_PREFIX,
    TOOLS_EXCLUDED_FROM_EVICTION,
    TOOL_RESULT_TOKEN_LIMIT,
    TOO_LARGE_TOOL_MSG,
    create_large_tool_result_preview,
    tool_result_char_budget,
)

__all__ = [
    "NUM_CHARS_PER_TOKEN",
    "OUTPUTS_LARGE_TOOL_RESULTS_PREFIX",
    "OUTPUTS_PREFIX",
    "TOOLS_EXCLUDED_FROM_EVICTION",
    "TOOL_RESULT_TOKEN_LIMIT",
    "TOO_LARGE_TOOL_MSG",
    "BackendProtocol",
    "EditResult",
    "FileInfo",
    "GlobResult",
    "GrepMatch",
    "GrepResult",
    "LsResult",
    "ReadResult",
    "WriteResult",
    "create_large_tool_result_preview",
    "tool_result_char_budget",
]


@dataclass
class FileInfo:
    path: str
    is_dir: bool = False
    size: int | None = None
    modified_at: str | None = None


@dataclass
class LsResult:
    entries: list[FileInfo] = field(default_factory=list)
    error: str | None = None


@dataclass
class ReadResult:
    content: str | None = None
    error: str | None = None
    total_lines: int | None = None


@dataclass
class GrepMatch:
    path: str
    line_number: int
    line: str


@dataclass
class GrepResult:
    matches: list[GrepMatch] = field(default_factory=list)
    error: str | None = None


@dataclass
class GlobResult:
    files: list[FileInfo] = field(default_factory=list)
    error: str | None = None


@dataclass
class WriteResult:
    path: str | None = None
    error: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    path: str | None = None
    occurrences: int = 0
    error: str | None = None


class BackendProtocol(abc.ABC):
    def resolve_path(self, session_id: str, agent_path: str) -> str | None:
        """Host filesystem path for ``agent_path`` when backed by disk; else ``None``."""
        return None

    @abc.abstractmethod
    def ls(self, session_id: str, path: str) -> LsResult:
        """List entries in a directory (agent path starting with /)."""

    @abc.abstractmethod
    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read text lines from file_path with 0-based line offset and max line count."""

    @abc.abstractmethod
    def grep(
        self,
        session_id: str,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for literal pattern in text files."""

    @abc.abstractmethod
    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        """Match file paths under path using glob pattern."""

    @abc.abstractmethod
    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        """Create or replace files. Implementations typically forbid replace except under ``/_outputs/``."""

    @abc.abstractmethod
    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace old_string with new_string; enforce uniqueness unless replace_all."""

    @abc.abstractmethod
    def execute(
        self,
        session_id: str,
        command: str,
        *,
        timeout: float = 120.0,
    ) -> str:
        """Run a shell command when supported; otherwise return a fixed error string.

        ``session_id`` is reserved for backends that scope execution per session; filesystem-only
        backends ignore it for ``cwd`` and use the host project root instead.
        Oversized output is not truncated here; tool middleware may spill to large_tool_results.
        """
