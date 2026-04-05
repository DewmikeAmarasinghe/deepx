from deepx.tools.workspace_tools import (
    write_file,
    read_file,
    append_to_file,
    edit_file,
    list_files,
)
from deepx.tools.planning_tools import write_todos, mark_done, read_todos
from deepx.tools.memory_tools import update_memory, read_memory

WORKSPACE_TOOLS = [write_file, read_file, append_to_file, edit_file, list_files]
PLANNING_TOOLS = [write_todos, mark_done, read_todos]
MEMORY_TOOLS = [update_memory, read_memory]
