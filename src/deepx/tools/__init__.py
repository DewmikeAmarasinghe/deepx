"""Deepx built-in tools: filesystem, memory, and planning."""

from __future__ import annotations

from agents.tool import Tool

from deepx.backends.local_shell import LocalShellBackend
from deepx.backends.protocol import BackendProtocol
from deepx.tools.agent_memory import save_memory
from deepx.tools.execute import execute
from deepx.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from deepx.tools.planning import think_tool, write_todos

FILESYSTEM_TOOLS: list[Tool] = [
    ls,
    read_file,
    write_file,
    edit_file,
    grep,
    glob,
]
LOCALSHELL_TOOLS: list[Tool] = [*FILESYSTEM_TOOLS, execute]
MEMORY_TOOLS = [save_memory]
PLANNING_TOOLS = [write_todos, think_tool]

EXECUTE_TOOL = execute
BUILTIN_TOOLS = [*LOCALSHELL_TOOLS, *MEMORY_TOOLS, *PLANNING_TOOLS]


def builtin_tools_for_backend(*, backend: BackendProtocol) -> list[Tool]:
    """Filesystem tools only, or filesystem + ``execute`` when using :class:`LocalShellBackend`."""
    if isinstance(backend, LocalShellBackend):
        return [*LOCALSHELL_TOOLS, *MEMORY_TOOLS, *PLANNING_TOOLS]
    return [*FILESYSTEM_TOOLS, *MEMORY_TOOLS, *PLANNING_TOOLS]


__all__ = [
    "BUILTIN_TOOLS",
    "builtin_tools_for_backend",
    "EXECUTE_TOOL",
    "FILESYSTEM_TOOLS",
    "LOCALSHELL_TOOLS",
    "MEMORY_TOOLS",
    "PLANNING_TOOLS",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "grep",
    "glob",
    "execute",
    "save_memory",
    "think_tool",
    "write_todos",
]
