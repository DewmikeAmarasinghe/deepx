from __future__ import annotations
import dataclasses
import json
import uuid
from datetime import datetime, timezone
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool
from agents.agent import Agent
from deepx.context import AgentContext
from deepx.middleware._utils import LARGE_OUTPUT_THRESHOLD, sanitize_path_component
from deepx.models import Plan


class WorkspaceHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(self, backend) -> None:
        self._backend = backend

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
        call_id = uuid.uuid4().hex[:12]
        session_id = context.context.session_id
        tool_name = sanitize_path_component(tool.name)
        is_large = len(result) > LARGE_OUTPUT_THRESHOLD
        saved_to = None

        if is_large:
            file_path = f"{tool_name}_{call_id}.txt"
            self._backend.write(session_id, file_path, result)
            saved_to = file_path

        log = {
            "call_id": call_id,
            "tool_name": tool.name,
            "agent_name": agent.name,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_chars": len(result),
            "output": result,
            "saved_to": saved_to,
        }
        self._backend.save_tool_log(session_id, log)


def wrap_tool_input_logging(tool: Tool, backend, session_id: str) -> Tool:
    from agents.tool import FunctionTool
    if not isinstance(tool, FunctionTool):
        return tool

    original_invoke = tool.on_invoke_tool

    async def logged_invoke(ctx, args_json: str) -> str:
        call_id = uuid.uuid4().hex[:12]
        result = await original_invoke(ctx, args_json)
        log = {
            "call_id": call_id,
            "tool_name": tool.name,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": json.loads(args_json) if args_json else {},
            "output_chars": len(str(result)),
            "output": str(result),
        }
        backend.save_tool_log(session_id, log)
        return result

    return dataclasses.replace(tool, on_invoke_tool=logged_invoke)