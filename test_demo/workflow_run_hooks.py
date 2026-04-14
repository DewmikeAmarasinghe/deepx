"""Run hooks that record discrete tool/LLM steps for workflow queries (demo / Chainlit)."""

from __future__ import annotations

from typing import Any

from agents.agent import Agent
from agents.items import ItemHelpers, ModelResponse
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool

from deepx.context import AgentContext


def _tool_label(tool: Tool) -> str:
    return str(getattr(tool, "name", None) or "?") or "?"


class WorkflowRunHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Appends JSON-serializable rows to ``event_log`` for UI polling.

    Compose with ``FilesystemHooks`` via ``compose_run_hooks`` on the runner; leave ``inner``
    unset so agent-start side effects are not duplicated.
    """

    def __init__(
        self,
        event_log: list[dict[str, Any]],
        *,
        inner: RunHooksBase[AgentContext, Agent[AgentContext]] | None = None,
    ) -> None:
        self._inner = inner
        self._event_log = event_log

    def _append(self, row: dict[str, Any]) -> None:
        self._event_log.append(row)
        if len(self._event_log) > 12_000:
            self._event_log[:] = self._event_log[-8000:]

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        if self._inner is not None:
            await self._inner.on_agent_start(context, agent)
        self._append({"kind": "agent", "name": agent.name})

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
    ) -> None:
        self._append(
            {
                "kind": "run_item",
                "name": "tool_called",
                "tool_name": _tool_label(tool),
                "tool_args": "{}",
                "agent_name": agent.name,
            }
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        text = str(result)
        preview = text[:4000] + ("…" if len(text) > 4000 else "")
        self._append(
            {
                "kind": "run_item",
                "name": "tool_output",
                "tool_name": _tool_label(tool),
                "output_preview": preview,
                "agent_name": agent.name,
            }
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        response: ModelResponse,
    ) -> None:
        chunks: list[str] = []
        for item in response.output[:8]:
            t = ItemHelpers.extract_text(item)
            if t:
                chunks.append(t[:1200])
        preview = "\n---\n".join(chunks)[:4000]
        if preview:
            self._append(
                {
                    "kind": "llm",
                    "agent_name": agent.name,
                    "output_preview": preview
                    + ("…" if len(preview) >= 4000 else ""),
                }
            )
