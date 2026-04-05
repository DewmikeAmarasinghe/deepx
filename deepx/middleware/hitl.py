import asyncio
import inspect

from agents import Agent, RunContextWrapper, RunHooks

from deepx.context import AgentContext


class HITLHooks(RunHooks[AgentContext]):
    def __init__(self, sensitive_tools: set[str], approval_fn=None):
        self.sensitive_tools = sensitive_tools
        self.approval_fn = approval_fn or self._cli_approval

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool,
    ):
        if tool.name not in self.sensitive_tools:
            return
        fn = self.approval_fn
        result = fn(tool.name, str(context))
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise Exception("rejected")

    async def _cli_approval(self, tool_name: str, ctx_str: str) -> bool:
        def _ask():
            return input(f"approve {tool_name}? [y/n]: ").strip().lower() == "y"

        return await asyncio.to_thread(_ask)
