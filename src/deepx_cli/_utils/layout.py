from __future__ import annotations

from rich.console import Console


def section_break(console: Console) -> None:
    """One blank line between major terminal blocks (thinking / tool / prose)."""
    console.print()
