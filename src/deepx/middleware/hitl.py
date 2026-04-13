from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable


class HumanInTheLoopHooks:
    def __init__(
        self,
        sensitive_tools: list[str],
        *,
        approval_fn: Callable[[str, str, str], bool | Awaitable[bool]] | None = None,
    ) -> None:
        self._sensitive = set(sensitive_tools)
        self._approved: set[str] = set()
        self._lock = asyncio.Lock()
        self._approval_fn = approval_fn or self._cli_approval

    @property
    def approval_fn(self) -> Callable[[str, str, str], bool | Awaitable[bool]]:
        return self._approval_fn

    @staticmethod
    def _cli_approval(agent_name: str, tool_name: str, tool_args_json: str) -> bool:
        _ = tool_args_json
        response = input(
            f"\n[HITL] Agent '{agent_name}' wants to call '{tool_name}'. Approve? [y/n]: "
        )
        return response.strip().lower() == "y"

    async def gate_tool(
        self, agent_name: str, tool_name: str, tool_args_json: str
    ) -> str | None:
        if tool_name not in self._sensitive:
            return None
        async with self._lock:
            if tool_name in self._approved:
                return None
            fn = self._approval_fn
            if inspect.iscoroutinefunction(fn):
                approved = await fn(agent_name, tool_name, tool_args_json)
            else:
                loop = asyncio.get_event_loop()
                approved = await loop.run_in_executor(
                    None,
                    lambda: fn(agent_name, tool_name, tool_args_json),
                )
            if not approved:
                return (
                    f"[Human-in-the-loop] The human declined approval for tool {tool_name!r}. "
                    "Do not retry this exact tool call without changing your approach or asking the user. "
                    "Use write_todos to update your plan and continue with other steps or tools."
                )
            self._approved.add(tool_name)
        return None
