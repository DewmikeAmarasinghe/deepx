from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel


def print_tool_call_panel(
    console: Console,
    agent_name: str,
    tool_name: str,
    arguments: Any,
    *,
    border_style: str = "cyan",
) -> None:
    """Bordered block for a tool invocation (name + JSON input), 3-space JSON indent."""
    try:
        body = json.dumps(arguments, indent=3, ensure_ascii=False)
    except TypeError:
        body = json.dumps(str(arguments), indent=3, ensure_ascii=False)
    title = f"{agent_name} · {tool_name}"
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_style,
            highlight=False,
        )
    )
