from agents import Agent, RunHooks
from agents.run_context import AgentHookContext

from deepx.context import AgentContext


class DeepRunHooks(RunHooks[AgentContext]):
    async def on_agent_end(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
        output,
    ):
        context.context.token_usage = context.usage.total_tokens


class MergedRunHooks(DeepRunHooks):
    def __init__(self, hitl):
        self._hitl = hitl

    async def on_tool_start(self, context, agent, tool):
        if self._hitl:
            await self._hitl.on_tool_start(context, agent, tool)
