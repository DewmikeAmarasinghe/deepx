from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeAlias, TypedDict

from agents.items import ToolApprovalItem
from agents.run_context import RunContextWrapper
from rich.console import Console

ApprovalChoice: TypeAlias = Literal["reject", "once", "always"]
ApprovalHandler: TypeAlias = Callable[[ToolApprovalItem], ApprovalChoice]


def iter_pending_tool_approvals(
    state: Any,
    interruptions: Iterable[ToolApprovalItem],
) -> list[ToolApprovalItem]:
    """Return interruption items that still need an explicit human decision."""
    ctx = getattr(state, "_context", None)
    out: list[ToolApprovalItem] = []
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
        out.append(item)
    return out


def apply_approval_choice(
    state: Any, item: ToolApprovalItem, choice: ApprovalChoice
) -> None:
    if choice == "reject":
        state.reject(item)
    elif choice == "once":
        state.approve(item, always_approve=False)
    else:
        state.approve(item, always_approve=True)


def apply_approval_handler(
    state: Any,
    interruptions: list[ToolApprovalItem],
    handler: ApprovalHandler,
) -> None:
    for item in iter_pending_tool_approvals(state, interruptions):
        apply_approval_choice(state, item, handler(item))


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
    """Terminal prompt from workflow query payload (or local CLI)."""
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
    "ApprovalChoice",
    "ApprovalHandler",
    "HitlPendingMeta",
    "apply_approval_choice",
    "apply_approval_handler",
    "apply_choices_to_state",
    "approval_choice",
    "approval_choice_from_meta",
    "iter_pending_tool_approvals",
]
