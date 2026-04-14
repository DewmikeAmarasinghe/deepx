from __future__ import annotations

import dataclasses
import uuid
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext
from agents.tool import FunctionTool, Tool

from deepx.backends.protocol import BackendProtocol
from deepx.backends.utils import (
    LARGE_TOOL_RESULTS_PREFIX,
    TOO_LARGE_TOOL_MSG,
    TOOLS_EXCLUDED_FROM_EVICTION,
    create_large_tool_result_preview,
    sanitize_tool_call_id,
    tool_result_char_budget,
    TOOL_RESULT_TOKEN_LIMIT,
)
from deepx.context import AgentContext
from deepx.middleware.hitl import HumanInTheLoopHooks, wrap_tools_for_hitl
from deepx.middleware.logs import run_log_load_plan, wrap_tools_for_logging
from deepx.tools.planning import Plan


def _tool_call_id(ctx: Any) -> str:
    tid = getattr(ctx, "tool_call_id", None)
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return "unknown_tool_call"


def _make_large_tool_results_invoke(
    original_invoke: Any,
    backend: BackendProtocol,
    *,
    tool_name: str,
    token_limit: int | None = TOOL_RESULT_TOKEN_LIMIT,
) -> Any:
    """Evict oversized tool returns to ``/large_tool_results/…`` (deepagents-style)."""

    budget = tool_result_char_budget(token_limit=token_limit)

    async def invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        if budget is None or tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            return result
        text = str(result)
        if len(text) <= budget:
            return result

        call_id = _tool_call_id(ctx)
        if call_id == "unknown_tool_call":
            safe = uuid.uuid4().hex
        else:
            safe = sanitize_tool_call_id(call_id)
        agent_path = f"{LARGE_TOOL_RESULTS_PREFIX}/{safe}"
        session_id = ctx.context.session_id
        wr = backend.write(session_id, agent_path, text)
        if wr.error:
            return result

        preview = create_large_tool_result_preview(text)
        return TOO_LARGE_TOOL_MSG.format(
            tool_call_id=call_id,
            file_path=agent_path,
            content_sample=preview,
        )

    return invoke


def wrap_tools_for_large_tool_results(
    tools: list[Tool],
    backend: BackendProtocol,
    *,
    token_limit: int | None = TOOL_RESULT_TOKEN_LIMIT,
) -> list[Tool]:
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            inv = _make_large_tool_results_invoke(
                tool.on_invoke_tool,
                backend,
                tool_name=tool.name,
                token_limit=token_limit,
            )
            out.append(dataclasses.replace(tool, on_invoke_tool=inv))
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
    wrapped = wrap_tools_for_large_tool_results(tools, backend)
    if debug:
        wrapped = wrap_tools_for_logging(wrapped, backend, agent_name)
    if hitl is not None:
        wrapped = wrap_tools_for_hitl(wrapped, hitl)
    return wrapped


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
        if not context.context.memory:
            from deepx.backends.filesystem import resolve_data_root

            dr = resolve_data_root(self._backend)
            if dr is not None:
                p = dr / "AGENTS.md"
                if p.is_file():
                    try:
                        context.context.memory = p.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        pass
