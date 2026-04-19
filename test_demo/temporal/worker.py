"""Temporal worker for durable orchestrator runs.

Run (from repo root, with ``uv sync --extra demo`` once)::

    temporal server start-dev

Terminal A::

    uv run --extra demo python -m test_demo.temporal.worker

Or use the Chainlit UI (``uv run chainlit run test_demo/ui/app.py``) with ``DEEPX_USE_TEMPORAL=true``,
which starts this workflow for each user message.

Prereqs: ``OPENAI_API_KEY`` for the worker process. The Agents SDK ``Runner.run`` executes inside an
activity (not the workflow sandbox) so SQLite sessions and threaded I/O work correctly.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from test_demo.temporal.activities import run_orchestrator_activity
from test_demo.temporal.workflows import TASK_QUEUE, DeepxOrchestratorWorkflow


async def _run_worker() -> None:
    load_dotenv()

    client = await Client.connect("localhost:7233", namespace="default")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DeepxOrchestratorWorkflow],
        activities=[run_orchestrator_activity],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    await worker.run()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
