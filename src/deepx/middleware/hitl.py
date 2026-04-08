from __future__ import annotations

import asyncio
from collections.abc import Callable

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import RunContextWrapper
from agents.tool import Tool

from deepx.context import AgentContext


class HumanInTheLoopHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Pauses before sensitive tool calls and prompts for human approval.

    Approval is remembered for the lifetime of the hooks instance — once a
    tool is approved it will not be asked again, even across parallel subagent
    runs that share this hooks object. An asyncio.Lock serializes the
    check-and-add so concurrent tool calls cannot both slip through before
    the first approval is recorded.

    Args:
        sensitive_tools: Tool names that require approval before execution.
        approval_fn: Optional override for the approval prompt.  Receives
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

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
    ) -> None:
        if tool.name not in self._sensitive:
            return
        async with self._lock:
            if tool.name in self._approved:
                return
            loop = asyncio.get_event_loop()
            approved = await loop.run_in_executor(
                None, self._approval_fn, agent.name, tool.name
            )
            if not approved:
                raise RuntimeError(
                    f"Human rejected tool call: {tool.name}. Do not retry this tool."
                )
            self._approved.add(tool.name)
