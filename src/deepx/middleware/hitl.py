from __future__ import annotations
from collections.abc import Callable
from agents.lifecycle import RunHooksBase
from agents.run_context import RunContextWrapper
from agents.tool import Tool
from agents.agent import Agent
from deepx.context import AgentContext


class HumanInTheLoopHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(
        self,
        sensitive_tools: set[str],
        approval_fn: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._sensitive = sensitive_tools
        self._approval_fn = approval_fn or self._cli_approval

    @staticmethod
    def _cli_approval(agent_name: str, tool_name: str) -> bool:
        response = input(f"\n[HITL] Agent '{agent_name}' wants to call '{tool_name}'. Approve? [y/n]: ")
        return response.strip().lower() == "y"

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
    ) -> None:
        if tool.name in self._sensitive:
            approved = self._approval_fn(agent.name, tool.name)
            if not approved:
                raise RuntimeError(f"Human rejected tool call: {tool.name}. Do not retry this tool.")