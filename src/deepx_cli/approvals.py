from __future__ import annotations

from typing import Any, TypedDict

from agents.items import ToolApprovalItem
from rich.console import Console

from deepx.hitl import ApprovalChoice, apply_approval_handler


class HitlPendingMeta(TypedDict, total=False):
    agent_name: str
    tool_name: str
    args_preview: str


def approval_choice(console: Console, item: ToolApprovalItem) -> ApprovalChoice:
    ag = getattr(item, "agent", None)
    agent_name = getattr(ag, "name", None) if ag is not None else None
    agent_name = agent_name or "agent"
    tool_name = item.tool_name or "tool"
    raw = getattr(item, "raw_item", None)
    args_preview = ""
    if raw is not None:
        args_preview = str(
            getattr(raw, "arguments", "") or getattr(raw, "args", "") or ""
        )
    return approval_choice_from_meta(
        console,
        {
            "agent_name": agent_name,
            "tool_name": tool_name,
            "args_preview": args_preview,
        },
    )


def approval_choice_from_meta(
    console: Console, meta: HitlPendingMeta
) -> ApprovalChoice:
    """Terminal prompt from workflow query payload (Temporal HITL pump)."""
    agent_name = meta.get("agent_name") or "agent"
    tool_name = meta.get("tool_name") or "tool"
    args_preview = meta.get("args_preview") or ""
    console.print()
    console.print(
        f"[yellow]{agent_name}:[/yellow] approve tool [bold]{tool_name}[/bold]"
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
    def _handler(item: ToolApprovalItem) -> ApprovalChoice:
        return approval_choice(console, item)

    apply_approval_handler(state, interruptions, _handler)


__all__ = [
    "HitlPendingMeta",
    "approval_choice",
    "approval_choice_from_meta",
    "apply_choices_to_state",
]
