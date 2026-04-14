from __future__ import annotations

NUM_CHARS_PER_TOKEN = 4

TOOL_RESULT_TOKEN_LIMIT = 20_000

LARGE_TOOL_RESULTS_PREFIX = "/large_tool_results"


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""Sanitize tool_call_id to prevent path traversal and separator issues.

    Replaces dangerous characters (., /, \\) with underscores.
    """
    return tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")


TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""


def create_large_tool_result_preview(
    content_str: str, *, head_lines: int = 5, tail_lines: int = 5
) -> str:
    """Head + tail preview with a middle truncation marker (deepagents-style)."""
    lines = content_str.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return "\n".join(line[:1000] for line in lines)

    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]
    omitted = len(lines) - head_lines - tail_lines
    return (
        "\n".join(head)
        + f"\n\n... [{omitted} lines truncated] ...\n\n"
        + "\n".join(tail)
    )


def tool_result_char_budget(*, token_limit: int | None = None) -> int | None:
    """Max characters before eviction; ``None`` means eviction disabled."""
    if token_limit is None:
        return None
    return NUM_CHARS_PER_TOKEN * int(token_limit)
