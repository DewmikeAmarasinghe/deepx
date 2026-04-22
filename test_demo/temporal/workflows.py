"""Temporal workflow: durable orchestrator via OpenAI Agents SDK + Temporal plugin.

``Runner.run`` is awaited **from workflow code** (with ``unsafe.imports_passed_through``), matching
the Temporal + Agents integration pattern: model and tool steps are scheduled as plugin-managed
activities and show up in Temporal history.

Tool approvals must **not** use blocking ``input()`` in workflow code. Pending approvals publish
state via :meth:`hitl_pending` query; the CLI sends choices with :meth:`hitl_approval` signal.

See https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from temporalio import workflow

TASK_QUEUE = "deepx-orchestrator-demo"


@dataclass
class DeepxOrchestratorInput:
    session_id: str
    prompt: str = ""
    resume: bool = False
    multi_turn: bool = False


@workflow.defn
class DeepxOrchestratorWorkflow:
    def __init__(self) -> None:
        self._hitl_choice: str | None = None
        self._hitl_pending: dict[str, str] | None = None
        self._inbox: list[str] = []
        self._done: bool = False
        self._output_seq: int = 0
        self._last_output: str = ""

    @workflow.signal
    def hitl_approval(self, choice: str) -> None:
        self._hitl_choice = choice

    @workflow.signal
    def user_message(self, text: str) -> None:
        self._inbox.append(text)

    @workflow.signal
    def end_session(self) -> None:
        self._done = True

    @workflow.query
    def hitl_pending(self) -> dict[str, str] | None:
        return self._hitl_pending

    @workflow.query
    def get_chat_status(self) -> dict[str, Any]:
        return {
            "seq": self._output_seq,
            "last_output": self._last_output,
        }

    def _normalize_hitl_choice(self, raw: str | None) -> Any:
        from deepx.hitl import ApprovalChoice

        c = (raw or "").strip().lower()
        if c in ("reject", "once", "always"):
            return cast(ApprovalChoice, c)
        return cast(ApprovalChoice, "reject")

    async def _resolve_hitl_signal(
        self,
        state: Any,
        interruptions: list[Any],
    ) -> None:
        from deepx.hitl import apply_approval_choice, iter_pending_tool_approvals

        for item in iter_pending_tool_approvals(state, interruptions):
            ag = getattr(item, "agent", None)
            agent_name = getattr(ag, "name", None) if ag is not None else None
            agent_name = agent_name or "agent"
            raw = getattr(item, "raw_item", None)
            args_preview = ""
            if raw is not None:
                args_preview = str(
                    getattr(raw, "arguments", "") or getattr(raw, "args", "") or ""
                )
            self._hitl_pending = {
                "agent_name": agent_name,
                "tool_name": item.tool_name or "",
                "args_preview": args_preview,
            }
            self._hitl_choice = None
            await workflow.wait_condition(lambda: self._hitl_choice is not None)
            choice = self._normalize_hitl_choice(self._hitl_choice)
            apply_approval_choice(state, item, choice)
        self._hitl_pending = None

    @workflow.run
    async def run(self, inp: DeepxOrchestratorInput) -> str:
        with workflow.unsafe.imports_passed_through():
            from rich.console import Console

            from deepx_cli.chat_stream import run_binding_until_settled
            from test_demo import orchestrator as orch

        runner = orch.orchestrator_runner_workflow

        async def resolve_approvals(
            state: Any,
            interrupts: list[Any],
            console: Any,
        ) -> None:
            _ = console
            await self._resolve_hitl_signal(state, interrupts)

        if not inp.multi_turn:
            binding = runner.bind(inp.session_id, resume=inp.resume)
            result = await run_binding_until_settled(
                binding,
                inp.prompt,
                console=Console(highlight=False),
                resolve_approvals=resolve_approvals,
            )
            return str(result.final_output)

        turn = 0
        if inp.prompt:
            self._inbox.append(inp.prompt)
        while not self._done:
            await workflow.wait_condition(
                lambda: len(self._inbox) > 0 or self._done,
            )
            if self._done:
                break
            text = self._inbox.pop(0)
            binding = runner.bind(inp.session_id, resume=(turn > 0))
            result = await run_binding_until_settled(
                binding,
                text,
                console=Console(highlight=False),
                resolve_approvals=resolve_approvals,
            )
            self._last_output = str(result.final_output)
            self._output_seq += 1
            turn += 1
        return self._last_output
