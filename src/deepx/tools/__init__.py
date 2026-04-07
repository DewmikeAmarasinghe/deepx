from deepx.tools.memory import read_memory, read_store, update_memory, write_store
from deepx.tools.planning import mark_done, read_todos, write_todos
from deepx.tools.workspace import (
    append_to_file,
    edit_file,
    execute,
    glob,
    grep,
    list_files,
    ls,
    read_file,
    write_file,
)

WORKSPACE_TOOLS = [
    ls,
    read_file,
    write_file,
    edit_file,
    append_to_file,
    glob,
    grep,
    list_files,
    execute,
]
PLANNING_TOOLS = [write_todos, mark_done, read_todos]
MEMORY_TOOLS = [update_memory, read_memory, read_store, write_store]

__all__ = [
    "WORKSPACE_TOOLS",
    "PLANNING_TOOLS",
    "MEMORY_TOOLS",
]
