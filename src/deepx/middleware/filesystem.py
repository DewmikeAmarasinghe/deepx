"""Run hooks tied to the filesystem-backed workspace (e.g. plan load from run logs)."""

from __future__ import annotations

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext

from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.logs import run_log_load_plan
from deepx.tools.planning import Plan


class FilesystemHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        context.context.agent_name = agent.name
        context.context.plan.agent_name = agent.name
        if context.context.resume:
            saved = run_log_load_plan(
                self._backend, context.context.session_id, agent.name
            )
            if saved:
                context.context.plan = Plan.model_validate_json(saved)


__all__ = ["FilesystemHooks"]
