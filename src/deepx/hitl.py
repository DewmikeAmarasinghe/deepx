"""Human-in-the-loop helpers for :class:`~agents.run_state.RunState` approvals.

The SDK stores pending tool approvals on :class:`~agents.result.RunResult` as
``interruptions`` (e.g. :class:`~agents.items.ToolApprovalItem`). Use these helpers so any UI
(terminal, web, Temporal signal) can drive ``state.approve`` / ``state.reject`` consistently.

Nested specialist runs mirror parent decisions via :func:`deepx.factory._subagent_tool_from_runner`
(peek / record path); that is separate from **human** resolution of outer interruptions handled here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeAlias

from agents.items import ToolApprovalItem
from agents.run_context import RunContextWrapper

ApprovalChoice: TypeAlias = Literal["reject", "once", "always"]
ApprovalHandler: TypeAlias = Callable[[ToolApprovalItem], ApprovalChoice]


def iter_pending_tool_approvals(
    state: Any,
    interruptions: Iterable[ToolApprovalItem],
) -> list[ToolApprovalItem]:
    """Return interruption items that still need an explicit human decision.

    Skips items already approved on the state's context (e.g. sticky allow for the run).
    """
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
    """Resolve each pending interruption using a callback (terminal, web, Temporal signal)."""
    for item in iter_pending_tool_approvals(state, interruptions):
        apply_approval_choice(state, item, handler(item))


__all__ = [
    "ApprovalChoice",
    "ApprovalHandler",
    "apply_approval_choice",
    "apply_approval_handler",
    "iter_pending_tool_approvals",
]
