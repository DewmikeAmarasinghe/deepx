"""Temporal workflow: OpenAI Agents SDK runs inside the workflow task (OpenAIAgentsPlugin).

Model and tool steps are recorded as Temporal activities by the plugin — do not wrap the full
orchestrator in a single custom activity. Stream snapshots are exposed via ``get_stream_events``
for the Chainlit demo (query polling).
"""

from __future__ import annotations

from dataclasses import dataclass

from temporalio import workflow

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
        """Serialized stream rows for UI tailing (same shape as ``serialize_stream_event``)."""
        return list(self._stream_events)

    @workflow.run
    async def run(self, inp: DeepxOrchestratorInput) -> str:
        with workflow.unsafe.imports_passed_through():
            from test_demo import orchestrator as orch
            from test_demo.temporal.stream_events import serialize_stream_event

            async def sink(ev: object) -> None:
                self._stream_events.append(serialize_stream_event(ev))
                if len(self._stream_events) > 12_000:
                    self._stream_events = self._stream_events[-8000:]

            runner = orch.build_orchestrator_runner(
                hitl_approval_fn=lambda *_a, **_k: True,
            )
            result = await runner.run_with_stream_sink(
                inp.prompt,
                session_id=inp.session_id,
                resume=False,
                stream_sink=sink,
            )
            self._stream_events.append(
                {"kind": "done", "output_preview": str(result.output)[:8000]},
            )
            return str(result.output)
