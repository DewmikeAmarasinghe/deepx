from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import FunctionTool, Tool

from deepx.backends.protocol import WorkspaceBackend
from deepx.context import AgentContext
from deepx.models import Plan

LARGE_OUTPUT_THRESHOLD = 80_000


class WorkspaceHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Lifecycle hooks that restore plan/memory state and handle large-output eviction.

    When *log_tools* is ``True``, tool calls are logged via
    :func:`wrap_tools_for_logging` which wraps each ``FunctionTool`` directly.
    This hook only writes a log entry for the large-output eviction path so
    that each tool call produces exactly **one** JSON file.
    """

    def __init__(self, backend: WorkspaceBackend, log_tools: bool) -> None:
        self._backend = backend
        self._log_tools = log_tools

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        if not context.context.memory:
            raw = self._backend.read_shared("AGENTS.md")
            if raw:
                context.context.memory = raw
        saved = self._backend.load_plan(context.context.session_id)
        if saved:
            context.context.plan = Plan.model_validate_json(saved)

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        if len(result) <= LARGE_OUTPUT_THRESHOLD:
            return
        session_id = context.context.session_id
        call_id = uuid.uuid4().hex[:12]
        file_path = f"{tool.name}_{call_id}.txt"
        self._backend.write(session_id, file_path, result)
        if self._log_tools:
            self._backend.save_tool_log(session_id, {
                "call_id": call_id,
                "tool_name": tool.name,
                "agent_name": agent.name,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_chars": len(result),
                "output": f"[evicted — full content saved to {file_path}]",
                "saved_to": file_path,
            })


def _make_logged_invoke(
    original_invoke: Any,
    tool_name: str,
    agent_name: str,
    backend: WorkspaceBackend,
    session_id: str,
) -> Any:
    async def logged_invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        backend.save_tool_log(session_id, {
            "call_id": uuid.uuid4().hex[:12],
            "tool_name": tool_name,
            "agent_name": agent_name,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": json.loads(args_json) if args_json else {},
            "output": str(result),
            "output_chars": len(str(result)),
        })
        return result
    return logged_invoke


def wrap_tools_for_logging(
    tools: list[Tool],
    backend: WorkspaceBackend,
    session_id: str,
    agent_name: str,
) -> list[Tool]:
    """Return copies of *tools* where every ``FunctionTool`` logs input + output."""
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            logged = _make_logged_invoke(
                tool.on_invoke_tool, tool.name, agent_name, backend, session_id
            )
            out.append(dataclasses.replace(tool, on_invoke_tool=logged))
        else:
            out.append(tool)
    return out