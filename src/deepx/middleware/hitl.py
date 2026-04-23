"""Human-in-the-loop types and tool wrapping (no imports from :mod:`deepx.context` at module level)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agents.tool import FunctionTool, Tool

DEFAULT_REJECTION_MESSAGE = (
    "[Human-in-the-loop] The human declined approval for this tool call. "
    "Do not retry the same call without changing inputs or asking the user. "
    "Use write_todos to adjust the plan and continue with other steps."
)


class HitlDecision(StrEnum):
    """Outcome of a host prompt for one gated tool invocation."""

    REJECT = "reject"
    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"


@dataclass(frozen=True, slots=True)
class HitlRequest:
    """Structured payload passed to the host when a gated tool is about to run."""

    session_id: str
    agent_name: str
    tool_name: str
    tool_call_id: str
    arguments_json: str


HitlCallback = Callable[[HitlRequest], Awaitable[HitlDecision]]


class Hitl:
    """Serializes gated tool prompts and records sticky allow-all per tool name (this binding)."""

    __slots__ = ("_policy", "_lock", "_always_allow_tools")

    def __init__(self, policy: HitlCallback | None = None) -> None:
        """If ``policy`` is ``None``, gated tools are auto-approved (tests / headless)."""
        self._policy = policy
        self._lock = asyncio.Lock()
        self._always_allow_tools: set[str] = set()

    @property
    def has_interactive_policy(self) -> bool:
        return self._policy is not None

    async def consult(self, request: HitlRequest) -> HitlDecision:
        """Resolve whether this invocation may proceed (possibly after prompting the host)."""
        async with self._lock:
            if request.tool_name in self._always_allow_tools:
                return HitlDecision.ALLOW_ONCE
            if self._policy is None:
                return HitlDecision.ALLOW_ONCE
            decision = await self._policy(request)
            if decision == HitlDecision.ALLOW_ALWAYS:
                self._always_allow_tools.add(request.tool_name)
            return decision


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

            agent_label = ac.agent_name or getattr(ctx, "agent", None) or ""
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


__all__ = [
    "DEFAULT_REJECTION_MESSAGE",
    "Hitl",
    "HitlCallback",
    "HitlDecision",
    "HitlRequest",
    "wrap_tools_for_hitl",
]
