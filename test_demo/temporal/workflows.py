"""Temporal workflow: OpenAI Agents SDK runs inside the workflow task (OpenAIAgentsPlugin).

Model and tool steps are recorded as Temporal activities by the plugin — do not wrap the full
orchestrator in a single custom activity. Discrete run snapshots are exposed via
``get_stream_events`` for the Chainlit demo (query polling; no token streaming in workflow).
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
        """Hook-fed rows for UI polling (tool boundaries, optional LLM summaries)."""
        return list(self._stream_events)

    @workflow.run
    async def run(self, inp: DeepxOrchestratorInput) -> str:
        with workflow.unsafe.imports_passed_through():
            from deepx.middleware.filesystem import FilesystemHooks
            from deepx.middleware.run_hooks import compose_run_hooks
            from test_demo import orchestrator as orch
            from test_demo.workflow_run_hooks import WorkflowRunHooks

            runner = orch.build_orchestrator_runner(
                hitl_approval_fn=lambda *_a, **_k: True,
            )
            wf_hooks = compose_run_hooks(
                FilesystemHooks(runner.backend),
                WorkflowRunHooks(self._stream_events),
            )
            result = await runner.run(
                inp.prompt,
                session_id=inp.session_id,
                resume=False,
                hooks=wf_hooks,
            )
            self._stream_events.append(
                {"kind": "done", "output_preview": str(result.output)[:8000]},
            )
            return str(result.output)
