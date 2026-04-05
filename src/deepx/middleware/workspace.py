from __future__ import annotations
import dataclasses
import json
import uuid
from datetime import datetime, timezone
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool, FunctionTool
from agents.agent import Agent
from deepx.context import AgentContext
from deepx.models import Plan


LARGE_OUTPUT_THRESHOLD = 80_000


class WorkspaceHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(self, backend, log_tools: bool) -> None:
        self._backend = backend
        self._log_tools = log_tools

    async def on_agent_start(
        self, context: AgentHookContext[AgentContext], agent: Agent[AgentContext]
    ) -> None:
        if not context.context.memory:
            raw = self._backend.read_shared("AGENTS.md")
            if raw:
                context.context.memory = raw

        saved_plan = self._backend.load_plan(context.context.session_id)
        if saved_plan:
            context.context.plan = Plan.model_validate_json(saved_plan)

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        session_id = context.context.session_id
        is_large = len(result) > LARGE_OUTPUT_THRESHOLD
        saved_to = None

        if is_large:
            file_path = f"{tool.name}_{uuid.uuid4().hex[:12]}.txt"
            self._backend.write(session_id, file_path, result)
            saved_to = file_path

        if self._log_tools:
            log = {
                "call_id": uuid.uuid4().hex[:12],
                "tool_name": tool.name,
                "agent_name": agent.name,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_chars": len(result),
                "output": result,
                "saved_to": saved_to,
            }
            self._backend.save_tool_log(session_id, log)


def _make_logged_invoke(original_invoke, tool_name: str, backend, session_id: str):
    async def logged_invoke(ctx, args_json: str):
        result = await original_invoke(ctx, args_json)
        log = {
            "call_id": uuid.uuid4().hex[:12],
            "tool_name": tool_name,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": json.loads(args_json) if args_json else {},
            "output": str(result),
            "output_chars": len(str(result)),
        }
        backend.save_tool_log(session_id, log)
        return result
    return logged_invoke


def wrap_tools_for_logging(tools: list, backend, session_id: str) -> list:
    result = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            logged = _make_logged_invoke(tool.on_invoke_tool, tool.name, backend, session_id)
            result.append(dataclasses.replace(tool, on_invoke_tool=logged))
        else:
            result.append(tool)
    return result