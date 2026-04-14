"""Temporal worker for durable orchestrator runs.

Run (from repo root, with ``uv sync --extra demo`` once)::

    temporal server start-dev

Terminal A::

    uv run --extra demo python -m test_demo.temporal.worker

Terminal B::

    uv run --extra demo python -m test_demo.temporal.start_workflow "Your prompt here"

Or use the Chainlit UI (``uv run chainlit run test_demo/ui/app.py``), which starts this workflow.

Prereqs: ``OPENAI_API_KEY`` for the worker process. Uses ``OpenAIAgentsPlugin`` per
https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from test_demo.temporal.workflows import TASK_QUEUE, DeepxOrchestratorWorkflow


async def _run_worker() -> None:
    load_dotenv()

    client = await Client.connect(
        "localhost:7233",
        namespace="default",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    start_to_close_timeout=timedelta(minutes=15),
                    retry_policy=RetryPolicy(maximum_attempts=5),
                ),
            ),
        ],
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DeepxOrchestratorWorkflow],
        activities=[],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    await worker.run()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
