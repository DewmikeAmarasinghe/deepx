from __future__ import annotations

import asyncio
from collections.abc import Callable


class HumanInTheLoopHooks:
    """Human approval for sensitive tools, enforced inside the tool invoke path.

    Approval is remembered for the lifetime of this instance — once a tool is
    approved it will not be asked again. An asyncio.Lock serializes the
    check-and-add so concurrent tool calls cannot both slip through before the
    first approval is recorded.

    If the human declines, ``gate_tool`` returns a rejection message (the model
    sees it as normal tool output) instead of raising.

    Args:
        sensitive_tools: Tool names that require approval before they run.
        approval_fn: Optional override for the approval prompt. Receives
            ``(agent_name, tool_name)`` and returns ``True`` to approve.
            Defaults to a CLI ``input()`` prompt.
    """

    def __init__(
        self,
        sensitive_tools: list[str],
        approval_fn: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._sensitive = set(sensitive_tools)
        self._approved: set[str] = set()
        self._lock = asyncio.Lock()
        self._approval_fn = approval_fn or self._cli_approval

    @staticmethod
    def _cli_approval(agent_name: str, tool_name: str) -> bool:
        response = input(
            f"\n[HITL] Agent '{agent_name}' wants to call '{tool_name}'. Approve? [y/n]: "
        )
        return response.strip().lower() == "y"

    async def gate_tool(self, agent_name: str, tool_name: str) -> str | None:
        """Return ``None`` if the tool may run; otherwise a rejection message for the model."""
        if tool_name not in self._sensitive:
            return None
        async with self._lock:
            if tool_name in self._approved:
                return None
            loop = asyncio.get_event_loop()
            approved = await loop.run_in_executor(
                None, self._approval_fn, agent_name, tool_name
            )
            if not approved:
                return (
                    f"[Human-in-the-loop] The human declined approval for tool {tool_name!r}. "
                    "Do not retry this exact tool call without changing your approach or asking the user. "
                    "Use write_todos to update your plan and continue with other steps or tools."
                )
            self._approved.add(tool_name)
        return None
