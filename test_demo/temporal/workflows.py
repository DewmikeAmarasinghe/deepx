"""Temporal workflow: durable orchestrator run via OpenAI Agents SDK + Temporal plugin.

Model calls execute as Temporal activities (``OpenAIAgentsPlugin``). The full agent run—including
SQLite-backed sessions and local backends—runs in a **normal activity** (see
``test_demo.temporal.activities``), not in workflow code, to avoid sandbox and asyncio executor
limitations.

See https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

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
            from test_demo.temporal.activities import (
                RunOrchestratorActivityInput,
                run_orchestrator_activity,
            )

        act_inp = RunOrchestratorActivityInput(
            prompt=inp.prompt,
            session_id=inp.session_id,
            resume=inp.resume,
        )
        return await workflow.execute_activity(
            run_orchestrator_activity,
            act_inp,
            start_to_close_timeout=timedelta(hours=2),
        )
