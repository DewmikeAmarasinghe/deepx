from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from deepx_cli._utils.layout import section_break


def print_hitl_approval_panel(
    console: Console,
    *,
    title: str,
    body: str,
    border_style: str = "yellow",
) -> None:
    section_break(console)
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_style,
            highlight=False,
        )
    )
