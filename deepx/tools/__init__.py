from deepx.tools.memory import read_memory, update_memory
from deepx.tools.planning import mark_done, read_todos, write_todos
from deepx.tools.shell import execute_command
from deepx.tools.vfs import append_to_file, edit_file, ls, read_file, write_file

CORE_TOOLS = [
    write_todos,
    mark_done,
    read_todos,
    write_file,
    read_file,
    edit_file,
    ls,
    append_to_file,
    update_memory,
    read_memory,
    execute_command,
]
