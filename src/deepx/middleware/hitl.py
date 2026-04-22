"""Wrap gated tools with :func:`wrap_tools_for_hitl` (imports :class:`deepx.hitl.Hitl` only at need)."""

from __future__ import annotations

import dataclasses
from typing import Any

from agents.tool import FunctionTool, Tool

from deepx.hitl import DEFAULT_REJECTION_MESSAGE, HitlDecision, HitlRequest


def wrap_tools_for_hitl(
    tools: list[Tool],
    interrupt_on: frozenset[str],
) -> list[Tool]:
    """Wrap ``FunctionTool`` instances whose names appear in ``interrupt_on``."""
    if not interrupt_on:
        return list(tools)

    out: list[Tool] = []
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            out.append(tool)
            continue
        if tool.name not in interrupt_on:
            out.append(tool)
            continue

        inner_invoke = tool.on_invoke_tool
        tool_name = tool.name

        async def hitl_invoke(
            ctx: Any,
            args_json: str,
            *,
            _inner: Any = inner_invoke,
            _tname: str = tool_name,
        ) -> Any:
            from deepx.context import AgentContext

            ac = getattr(ctx, "context", None)
            if not isinstance(ac, AgentContext):
                return await _inner(ctx, args_json)

            hitl = ac.hitl
            if hitl is None:
                return await _inner(ctx, args_json)

            call_id = getattr(ctx, "tool_call_id", None)
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = "unknown"

            agent_label = (ac.agent_name or getattr(ctx, "agent", None) or "")
            if hasattr(agent_label, "name"):
                agent_label = getattr(agent_label, "name", "") or str(agent_label)
            agent_label = str(agent_label).strip() or "agent"

            req = HitlRequest(
                session_id=ac.session_id,
                agent_name=agent_label,
                tool_name=_tname,
                tool_call_id=call_id.strip(),
                arguments_json=args_json or "",
            )
            decision = await hitl.consult(req)
            if decision == HitlDecision.REJECT:
                return DEFAULT_REJECTION_MESSAGE
            return await _inner(ctx, args_json)

        out.append(dataclasses.replace(tool, on_invoke_tool=hitl_invoke))

    return out


__all__ = ["wrap_tools_for_hitl"]
