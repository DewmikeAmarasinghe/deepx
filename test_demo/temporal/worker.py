"""Temporal worker for durable orchestrator runs.

Run (from repo root, with ``uv sync --extra demo`` once)::

    temporal server start-dev

Terminal A::

    uv run --extra demo python -m test_demo.temporal.worker

Set ``USE_TEMPORAL=true`` when using ``python -m test_demo.orchestrator`` so the CLI starts
workflows instead of calling ``Runner`` locally.

The agent loop runs **inside the workflow** (see ``workflows.py``), so ``OpenAIAgentsPlugin`` can
record model/tool work as Temporal activities. Worker process defaults ``LANGSMITH_TRACING`` off
when unset (set it to ``true`` in the environment to enable LangSmith in the worker).

See ``test_demo/temporal/README.md`` for architecture notes.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from test_demo.temporal.client import connect_temporal_client
from test_demo.temporal.workflows import TASK_QUEUE, DeepxOrchestratorWorkflow


async def _run_worker() -> None:
    load_dotenv()
    if "LANGSMITH_TRACING" not in os.environ:
        os.environ["LANGSMITH_TRACING"] = "false"

    client = await connect_temporal_client()
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DeepxOrchestratorWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    await worker.run()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
