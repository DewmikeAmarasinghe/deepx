from __future__ import annotations

from typing import Any

from agents.items import ToolApprovalItem
from agents.run_context import RunContextWrapper
from rich.console import Console


def approval_choice(console: Console, item: ToolApprovalItem) -> str:
    ag = getattr(item, "agent", None)
    agent_name = getattr(ag, "name", None) if ag is not None else None
    agent_name = agent_name or "agent"
    raw = getattr(item, "raw_item", None)
    args_preview = ""
    if raw is not None:
        args_preview = str(
            getattr(raw, "arguments", "") or getattr(raw, "args", "") or ""
        )[:800]
    console.print()
    console.print(
        f"[yellow]Tool approval[/yellow]  agent=[bold]{agent_name}[/bold]  "
        f"tool=[bold]{item.tool_name}[/bold]"
    )
    if args_preview:
        console.print(f"[dim]{args_preview}[/dim]")
    console.print("  [1] Reject")
    console.print("  [2] Allow once (this call only)")
    console.print("  [3] Allow for the rest of this run")
    while True:
        choice = input("Choice [1-3]: ").strip().lower()
        if choice in ("1", "r", "reject", "n", "no"):
            return "reject"
        if choice in ("2", "o", "once", "y", "yes"):
            return "once"
        if choice in ("3", "a", "always", "session"):
            return "always"
        console.print("Enter 1, 2, or 3.")


def apply_choices_to_state(
    state: Any, interruptions: list[ToolApprovalItem], console: Console
) -> None:
    ctx = getattr(state, "_context", None)
    for item in list(interruptions):
        if ctx is not None:
            call_id = RunContextWrapper._resolve_call_id(item) or ""
            pre = ctx.get_approval_status(
                item.tool_name or "",
                call_id,
                existing_pending=item,
            )
            if pre is True:
                continue
        choice = approval_choice(console, item)
        if choice == "reject":
            state.reject(item)
        elif choice == "once":
            state.approve(item, always_approve=False)
        else:
            state.approve(item, always_approve=True)


__all__ = ["approval_choice", "apply_choices_to_state"]
