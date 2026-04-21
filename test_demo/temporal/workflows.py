"""Temporal workflow: durable orchestrator via OpenAI Agents SDK + Temporal plugin.

``Runner.run`` is awaited **from workflow code** (with ``unsafe.imports_passed_through``), matching
the Temporal + Agents integration pattern: model and tool steps are scheduled as plugin-managed
activities and show up in Temporal history—not hidden inside a single custom activity.

Uses ``UnsandboxedWorkflowRunner`` on the worker (see ``worker.py``) so workflow code can import
application modules. Human-in-the-loop (stdin approvals) still runs in the worker process hosting
the workflow task; for production, prefer signals/updates instead of blocking ``input()``.

See https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python
"""

from __future__ import annotations

from dataclasses import dataclass

from temporalio import workflow

TASK_QUEUE = "deepx-orchestrator-demo"


@dataclass
class DeepxOrchestratorInput:
    prompt: str
    session_id: str
    resume: bool = False


@workflow.defn
class DeepxOrchestratorWorkflow:
    @workflow.run
    async def run(self, inp: DeepxOrchestratorInput) -> str:
        with workflow.unsafe.imports_passed_through():
            from rich.console import Console

            from deepx_cli.chat_stream import run_binding_until_settled
            from test_demo import orchestrator as orch

        runner = orch.orchestrator_runner_workflow
        binding = runner.bind(inp.session_id, resume=inp.resume)
        result = await run_binding_until_settled(
            binding,
            inp.prompt,
            console=Console(highlight=False),
        )
        return str(result.final_output)
