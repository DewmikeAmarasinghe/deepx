"""Temporal activities: full Agents SDK run (outside workflow sandbox)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from temporalio import activity
from temporalio.client import Client

from agents.agent import Agent
from agents.items import ItemHelpers, ModelResponse
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool

from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks
from deepx.middleware.run_hooks import compose_run_hooks

STREAM_SIGNAL = "append_stream_events"


def _temporal_hitl_approval_fn() -> Any:
    """HITL for worker-only runs: auto when DEEPX_HITL_AUTO_APPROVE=1, else stdin."""
    from deepx.middleware.hitl import HumanInTheLoopHooks

    def _fn(agent_name: str, tool_name: str, tool_args_json: str) -> bool:
        if os.environ.get("DEEPX_HITL_AUTO_APPROVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return True
        return HumanInTheLoopHooks._cli_approval(agent_name, tool_name, tool_args_json)

    return _fn


def _tool_label(tool: Tool) -> str:
    return str(getattr(tool, "name", None) or "?") or "?"


class _SignalBatchedRunHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Forwards hook events to the parent workflow via signals (batched)."""

    def __init__(
        self,
        client: Client,
        workflow_id: str,
        *,
        inner: RunHooksBase[AgentContext, Agent[AgentContext]] | None = None,
        batch_size: int = 12,
    ) -> None:
        self._inner = inner
        self._client = client
        self._workflow_id = workflow_id
        self._batch_size = max(1, batch_size)
        self._buf: list[dict[str, Any]] = []

    async def _flush(self) -> None:
        if not self._buf:
            return
        batch = self._buf[:]
        self._buf.clear()
        handle = self._client.get_workflow_handle(self._workflow_id)
        await handle.signal(STREAM_SIGNAL, batch)

    async def _queue(self, row: dict[str, Any]) -> None:
        self._buf.append(row)
        if len(self._buf) >= self._batch_size:
            await self._flush()

    async def close(self) -> None:
        await self._flush()

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        if self._inner is not None:
            await self._inner.on_agent_start(context, agent)
        await self._queue({"kind": "agent", "name": agent.name})

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
    ) -> None:
        if self._inner is not None:
            await self._inner.on_tool_start(context, agent, tool)
        await self._queue(
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
        if self._inner is not None:
            await self._inner.on_tool_end(context, agent, tool, result)
        text = str(result)
        preview = text[:4000] + ("…" if len(text) > 4000 else "")
        await self._queue(
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
        if self._inner is not None:
            await self._inner.on_llm_end(context, agent, response)
        chunks: list[str] = []
        for item in response.output[:8]:
            t = ItemHelpers.extract_text(item)
            if t:
                chunks.append(t[:1200])
        preview = "\n---\n".join(chunks)[:4000]
        if preview:
            await self._queue(
                {
                    "kind": "llm",
                    "agent_name": agent.name,
                    "output_preview": preview
                    + ("…" if len(preview) >= 4000 else ""),
                }
            )


@dataclass
class DeepxOrchestratorActivityInput:
    prompt: str
    session_id: str


@activity.defn(name="run_orchestrator_activity")
async def run_orchestrator_activity(inp: DeepxOrchestratorActivityInput) -> str:
    info = activity.info()
    wf_id = info.workflow_id
    if not wf_id:
        msg = "run_orchestrator_activity requires a parent workflow_id"
        raise RuntimeError(msg)

    from test_demo.temporal.client import connect_temporal_client

    client = await connect_temporal_client()
    signal_hooks = _SignalBatchedRunHooks(client, wf_id)

    from test_demo import orchestrator as orch

    runner = orch.build_orchestrator_runner(
        hitl_approval_fn=_temporal_hitl_approval_fn(),
    )
    hooks = compose_run_hooks(FilesystemHooks(runner.backend), signal_hooks)
    try:
        result = await runner.run(
            inp.prompt,
            session_id=inp.session_id,
            resume=False,
            hooks=hooks,
        )
    finally:
        await signal_hooks.close()

    return str(result.output)
