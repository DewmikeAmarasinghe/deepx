"""Core human-in-the-loop types (no agent or middleware imports — safe for :class:`AgentContext`)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

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


__all__ = [
    "DEFAULT_REJECTION_MESSAGE",
    "Hitl",
    "HitlCallback",
    "HitlDecision",
    "HitlRequest",
]
