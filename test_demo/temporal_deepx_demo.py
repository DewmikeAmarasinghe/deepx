"""Temporal **worker** for durable orchestrator runs (optional profile: ``uv sync --extra demo``).

Run the worker (from repo root)::

  uv run --extra demo python test_demo/temporal_deepx_demo.py

In another terminal, start a workflow with the Temporal CLI (example; adjust ``session_id``)::

  temporal workflow start \\
    --task-queue deepx-orchestrator-demo \\
    --type DeepxOrchestratorWorkflow \\
    --input '{{"prompt":"Say hello in one sentence.","session_id":"'$(uuidgen | head -c12)'"}}'

Or run the orchestrator interactively without Temporal: ``python test_demo/orchestrator.py``.

Prereqs: ``temporal server start-dev``, ``OPENAI_API_KEY`` in the environment for the worker process.

Uses ``OpenAIAgentsPlugin`` on the worker client per Temporal’s OpenAI Agents integration; see
https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TASK_QUEUE = "deepx-orchestrator-demo"


@dataclass
class DeepxOrchestratorInput:
    prompt: str
    session_id: str


@activity.defn
async def run_deepx_orchestrator_activity(inp: DeepxOrchestratorInput) -> str:
    """Runs one full Deepx orchestrator turn (may call subagents and tools)."""
    from test_demo import orchestrator as orch

    result = await orch.agent.run(inp.prompt, session_id=inp.session_id)
    return str(result.output)


@workflow.defn
class DeepxOrchestratorWorkflow:
    @workflow.run
    async def run(self, inp: DeepxOrchestratorInput) -> str:
        return await workflow.execute_activity(
            run_deepx_orchestrator_activity,
            inp,
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=5),
        )


async def _run_worker() -> None:
    from temporalio.client import Client
    from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
    from temporalio.worker import Worker

    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    start_to_close_timeout=timedelta(minutes=15),
                ),
            ),
        ],
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DeepxOrchestratorWorkflow],
        activities=[run_deepx_orchestrator_activity],
    )
    await worker.run()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
