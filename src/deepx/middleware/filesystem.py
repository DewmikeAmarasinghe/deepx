from __future__ import annotations

import dataclasses
import uuid
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext
from agents.tool import FunctionTool, Tool

from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.logs import wrap_tools_for_logging
from deepx.models import Plan

LARGE_OUTPUT_THRESHOLD = 40_000


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
            saved = self._backend.load_plan(context.context.session_id, agent.name)
            if saved:
                context.context.plan = Plan.model_validate_json(saved)
        if not context.context.memory:
            rr = self._backend.read(
                context.context.session_id, "/store/AGENTS.md", 0, 1_000_000
            )
            if rr.content is not None and not rr.error:
                context.context.memory = rr.content


def _make_evicting_invoke(original_invoke: Any, backend: BackendProtocol) -> Any:
    async def invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        text = str(result)
        if len(text) <= LARGE_OUTPUT_THRESHOLD:
            return result
        session_id = ctx.context.session_id

        call_id = uuid.uuid4().hex[:12]
        dest = f"/_workspace_/large_tool_results/{call_id}.txt"
        wr = backend.write(session_id, dest, text)
        if wr.error:
            return result
        preview = "\n".join(text.splitlines()[:10])
        return (
            f"[Output was large and saved to {dest}. Use read_file to access it. "
            f"Preview:\n{preview}]"
        )

    return invoke


def wrap_tools_with_large_output_eviction(
    tools: list[Tool],
    backend: BackendProtocol,
) -> list[Tool]:
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            inv = _make_evicting_invoke(tool.on_invoke_tool, backend)
            out.append(dataclasses.replace(tool, on_invoke_tool=inv))
        else:
            out.append(tool)
    return out


def wrap_tools_for_hitl(
    tools: list[Tool],
    hitl: HumanInTheLoopHooks,
) -> list[Tool]:
    """Wrap sensitive FunctionTools so approval runs before invoke; declines become tool output."""
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool) and tool.name in hitl._sensitive:
            inner = tool.on_invoke_tool
            name = tool.name

            async def hitl_invoke(
                ctx: Any,
                args_json: str,
                *,
                _inner: Any = inner,
                _hitl: HumanInTheLoopHooks = hitl,
                _name: str = name,
            ) -> Any:
                agent_name = getattr(ctx.context, "agent_name", "") or "agent"
                msg = await _hitl.gate_tool(agent_name, _name, args_json)
                if msg is not None:
                    return msg
                return await _inner(ctx, args_json)

            out.append(dataclasses.replace(tool, on_invoke_tool=hitl_invoke))
        else:
            out.append(tool)
    return out


def apply_tool_pipeline(
    tools: list[Tool],
    backend: BackendProtocol,
    *,
    agent_name: str,
    debug: bool,
    hitl: HumanInTheLoopHooks | None = None,
) -> list[Tool]:
    wrapped = wrap_tools_with_large_output_eviction(tools, backend)
    if debug:
        wrapped = wrap_tools_for_logging(wrapped, backend, agent_name)
    if hitl is not None:
        wrapped = wrap_tools_for_hitl(wrapped, hitl)
    return wrapped
