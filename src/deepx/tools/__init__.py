"""Deepx built-in tools: filesystem, memory, and planning."""

from __future__ import annotations

from deepx.tools.agent_memory import save_memory
from deepx.tools.execute import execute
from deepx.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from deepx.tools.planning import think_tool, write_todos

FILESYSTEM_TOOLS = [ls, read_file, write_file, edit_file, grep, glob, execute]
MEMORY_TOOLS = [save_memory]
PLANNING_TOOLS = [write_todos, think_tool]

BUILTIN_TOOLS = [*FILESYSTEM_TOOLS, *MEMORY_TOOLS, *PLANNING_TOOLS]

__all__ = [
    "BUILTIN_TOOLS",
    "FILESYSTEM_TOOLS",
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
