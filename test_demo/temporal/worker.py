"""Temporal worker for durable orchestrator runs.

Run (from repo root, with ``uv sync --extra demo`` once)::

    temporal server start-dev

Terminal A::

    uv run --extra demo python -m test_demo.temporal.worker

Set ``USE_TEMPORAL=true`` when using ``python -m test_demo.orchestrator`` so the CLI starts
workflows instead of calling ``Runner`` locally.

Prereqs: ``OPENAI_API_KEY`` for the worker process. ``OpenAIAgentsPlugin`` (on ``Client.connect``)
auto-registers **model** activities for LLM calls. **Application** activities—such as
``run_orchestrator_activity``, which hosts SQLite and blocking agent work—must still be registered
explicitly on the worker.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from test_demo.temporal.activities import run_orchestrator_activity
from test_demo.temporal.client import connect_temporal_client
from test_demo.temporal.workflows import TASK_QUEUE, DeepxOrchestratorWorkflow


async def _run_worker() -> None:
    load_dotenv()

    client = await connect_temporal_client()
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
