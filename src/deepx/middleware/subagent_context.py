"""Isolate :class:`~deepx.context.AgentContext` for nested subagent runs (SDK ``Agent.as_tool``).

The parent and child share the same :class:`~agents.run_context.RunContextWrapper` during a nested
run; without isolation, ``plan``, ``skills``, and ``is_subagent`` would leak between agents.
"""

from __future__ import annotations

from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext

from deepx.context import AgentContext


class SubagentContextIsolationHook(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Swap in a fresh :class:`AgentContext` while a named subagent runs; restore on end."""

    def __init__(self, subagent_name: str, skills_markdown: str) -> None:
        self._subagent_name = subagent_name
        self._skills_markdown = skills_markdown
        self._saved: list[AgentContext] = []

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        if agent.name != self._subagent_name:
            return
        parent = context.context
        self._saved.append(parent)
        context.context = AgentContext(
            session_id=parent.session_id,
            backend=parent.backend,
            agent_name=agent.name,
            memory=parent.memory,
            skills=self._skills_markdown,
            debug=parent.debug,
            resume=False,
            is_subagent=True,
        )

    async def on_agent_end(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
        output: Any,
    ) -> None:
        if agent.name != self._subagent_name or not self._saved:
            return
        context.context = self._saved.pop()
