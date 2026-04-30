from __future__ import annotations

from rich.console import Console

from deepx_cli._utils.layout import section_break


def print_assistant_prose_break(console: Console) -> None:
    """Extra separation before the next assistant `agent:` line and streamed prose."""
    section_break(console)
