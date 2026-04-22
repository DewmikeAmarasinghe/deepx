from __future__ import annotations

import dataclasses
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext
from agents.tool import FunctionTool, Tool

from deepx.backends.filesystem import (
    OUTPUTS_LARGE_TOOL_RESULTS_PREFIX,
    TOO_LARGE_TOOL_MSG,
    TOOL_RESULT_TOKEN_LIMIT,
    TOOLS_EXCLUDED_FROM_EVICTION,
    create_large_tool_result_preview,
    sanitize_tool_call_id,
    tool_result_char_budget,
)
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.hitl import wrap_tools_for_hitl
from deepx.middleware.logs import run_log_load_plan, wrap_tools_for_logging
from deepx.tools.planning import Plan


def _tool_call_id(ctx: Any) -> str:
    tid = getattr(ctx, "tool_call_id", None)
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return "unknown_tool_call"


def _readable_large_tool_agent_path(tool_name: str, tool_call_id: str) -> str:
    base = (
        re.sub(r"[^a-zA-Z0-9_-]+", "_", (tool_name or "tool").strip()).strip("_")[:48]
        or "tool"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if tool_call_id == "unknown_tool_call":
        suffix = uuid.uuid4().hex[:8]
    else:
        suffix = sanitize_tool_call_id(tool_call_id)[:32]
    return f"{OUTPUTS_LARGE_TOOL_RESULTS_PREFIX}/{base}_{stamp}_{suffix}.txt"


def _make_large_tool_results_invoke(
    original_invoke: Any,
    backend: BackendProtocol,
    *,
    tool_name: str,
    token_limit: int | None = TOOL_RESULT_TOKEN_LIMIT,
) -> Any:
    """Evict oversized tool returns under ``/_outputs/large_tool_results/``."""

    budget = tool_result_char_budget(token_limit=token_limit)

    async def invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        if budget is None or tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            return result
        text = str(result)
        if len(text) <= budget:
            return result

        call_id = _tool_call_id(ctx)
        agent_path = _readable_large_tool_agent_path(tool_name, call_id)
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
    interrupt_on: frozenset[str] | None = None,
) -> list[Tool]:
    wrapped = wrap_tools_for_large_tool_results(tools, backend)
    if debug:
        wrapped = wrap_tools_for_logging(wrapped, backend, agent_name)
    wrapped = wrap_tools_for_hitl(wrapped, interrupt_on or frozenset())
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
