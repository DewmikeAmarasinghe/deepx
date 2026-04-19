"""Temporal workflow: delegates the Agents SDK run to an activity (avoids workflow sandbox limits).

Stream rows are sent from the activity via signals and exposed through ``get_stream_events`` for
Chainlit polling. See https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

TASK_QUEUE = "deepx-orchestrator-demo"


@dataclass
class DeepxOrchestratorInput:
    prompt: str
    session_id: str


@workflow.defn
class DeepxOrchestratorWorkflow:
    def __init__(self) -> None:
        self._stream_events: list[dict] = []

    @workflow.query
    def get_stream_events(self) -> list[dict]:
        """Hook-fed rows for UI polling (tool boundaries, optional LLM summaries)."""
        return list(self._stream_events)

    @workflow.signal
    def append_stream_events(self, rows: list[dict]) -> None:
        """Activity → workflow: batched stream rows for Chainlit."""
        for row in rows:
            if isinstance(row, dict):
                self._stream_events.append(row)
        if len(self._stream_events) > 12_000:
            self._stream_events[:] = self._stream_events[-8000:]

    @workflow.run
    async def run(self, inp: DeepxOrchestratorInput) -> str:
        with workflow.unsafe.imports_passed_through():
            from test_demo.temporal.activities import (
                DeepxOrchestratorActivityInput,
                run_orchestrator_activity,
            )

        act_inp = DeepxOrchestratorActivityInput(
            prompt=inp.prompt,
            session_id=inp.session_id,
        )
        result = await workflow.execute_activity(
            run_orchestrator_activity,
            act_inp,
            start_to_close_timeout=timedelta(minutes=45),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        self._stream_events.append(
            {"kind": "done", "output_preview": str(result)[:8000]},
        )
        return str(result)
