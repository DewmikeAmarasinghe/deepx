"""Human-in-the-loop types and tool wrapping (no imports from :mod:`deepx.context` at module level)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agents.tool import FunctionTool, Tool

from deepx.backends.protocol import BackendProtocol
from deepx.backends.utils import MAX_READ_FILE_LINES, data_root_as_agent_path

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


def _approvals_rel(session_id: str) -> str:
    return f"sessions/{session_id}/approvals.json"


def _merge_allow_always_payload(
    into: set[tuple[str, str]], content: str
) -> None:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return
    entries = data.get("allow_always")
    if not isinstance(entries, list):
        return
    for item in entries:
        if not isinstance(item, dict):
            continue
        an = str(item.get("agent_name", "")).strip()
        tn = str(item.get("tool_name", "")).strip()
        if an and tn:
            into.add((an, tn))


class Hitl:
    """Serializes gated tool prompts; ``allow_always`` is per (agent_name, tool_name) and persisted.

    Persist/load uses the **tool runner's** :class:`~deepx.backends.protocol.BackendProtocol` (the agent
    that owns the gated tool), not only the top-level runner's backend — so e.g. ``execute`` on
    ``web_agent`` writes ``/.deepx/sessions/<id>/approvals.json`` via that agent's backend.
    """

    __slots__ = (
        "_policy",
        "_lock",
        "_always_allow",
        "_backend",
        "_session_id",
        "_merged_backend_ids",
    )

    def __init__(self, policy: HitlCallback | None = None) -> None:
        """If ``policy`` is ``None``, gated tools are auto-approved (tests / headless)."""
        self._policy = policy
        self._lock = asyncio.Lock()
        self._always_allow: set[tuple[str, str]] = set()
        self._backend: BackendProtocol | None = None
        self._session_id: str | None = None
        self._merged_backend_ids: set[int] = set()

    def attach_session(self, backend: BackendProtocol, session_id: str) -> None:
        """Prime session id and load approvals from the **bound** runner's backend if session changed."""
        sid = (session_id or "").strip() or None
        session_changed = sid != self._session_id
        backend_changed = backend is not self._backend
        self._backend = backend
        self._session_id = sid
        if sid is None:
            self._always_allow.clear()
            self._merged_backend_ids.clear()
            return
        if session_changed or backend_changed:
            self._always_allow.clear()
            self._merged_backend_ids.clear()
            self._merge_backend_allowlist(sid, backend)
            self._merged_backend_ids.add(id(backend))

    def _merge_backend_allowlist(
        self, session_id: str, backend: BackendProtocol
    ) -> None:
        path = data_root_as_agent_path(_approvals_rel(session_id))
        rr = backend.read(session_id, path, 0, MAX_READ_FILE_LINES)
        if rr.error or not rr.content:
            return
        _merge_allow_always_payload(self._always_allow, rr.content)

    def _persist(self, backend: BackendProtocol) -> None:
        if not self._session_id:
            return
        entries = [
            {"agent_name": a, "tool_name": t}
            for a, t in sorted(self._always_allow, key=lambda x: (x[0], x[1]))
        ]
        payload = json.dumps({"allow_always": entries}, indent=2)
        path = data_root_as_agent_path(_approvals_rel(self._session_id))
        wr = backend.write(self._session_id, path, payload)
        if wr.error:
            raise OSError(wr.error)

    @property
    def has_interactive_policy(self) -> bool:
        return self._policy is not None

    async def consult(
        self,
        request: HitlRequest,
        *,
        tool_backend: BackendProtocol | None = None,
    ) -> HitlDecision:
        """Resolve whether this invocation may proceed (possibly after prompting the host)."""
        b = tool_backend or self._backend
        async with self._lock:
            sid = (request.session_id or "").strip() or self._session_id
            if sid and b is not None and id(b) not in self._merged_backend_ids:
                self._merge_backend_allowlist(sid, b)
                self._merged_backend_ids.add(id(b))
            key = (request.agent_name, request.tool_name)
            if key in self._always_allow:
                return HitlDecision.ALLOW_ONCE
            if self._policy is None:
                return HitlDecision.ALLOW_ONCE
            decision = await self._policy(request)
            if decision == HitlDecision.ALLOW_ALWAYS:
                self._always_allow.add(key)
                if b is None:
                    raise OSError(
                        "HITL allow-always persist requires tool_backend or attach_session backend"
                    )
                self._persist(b)
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
            decision = await hitl.consult(req, tool_backend=ac.backend)
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
