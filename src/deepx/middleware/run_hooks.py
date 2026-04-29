"""Compose multiple ``RunHooksBase`` instances (OpenAI Agents analogue of stacked middleware)."""

from __future__ import annotations

from typing import Any, Sequence

from agents.agent import Agent
from agents.items import ModelResponse, TResponseInputItem
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool

from deepx.context import AgentContext


def compose_run_hooks(
    *hooks: RunHooksBase[AgentContext, Agent[AgentContext]],
) -> RunHooksBase[AgentContext, Agent[AgentContext]]:
    """Return a single hook object that forwards each callback to ``hooks`` in order."""
    hs = tuple(hooks)
    if len(hs) == 1:
        return hs[0]
    return ChainedRunHooks(hs)


class ChainedRunHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(
        self, hooks: Sequence[RunHooksBase[AgentContext, Agent[AgentContext]]]
    ) -> None:
        self._hooks = tuple(hooks)

    async def on_llm_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        for h in self._hooks:
            await h.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        response: ModelResponse,
    ) -> None:
        for h in self._hooks:
            await h.on_llm_end(context, agent, response)

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        for h in self._hooks:
            await h.on_agent_start(context, agent)

    async def on_agent_end(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
        output: Any,
    ) -> None:
        for h in self._hooks:
            await h.on_agent_end(context, agent, output)

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
    ) -> None:
        for h in self._hooks:
            await h.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        for h in self._hooks:
            await h.on_tool_end(context, agent, tool, result)
