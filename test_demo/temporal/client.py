"""Temporal client helpers for the orchestrator demo."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from temporalio.client import Client, WorkflowHandle
from temporalio.common import RetryPolicy
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin

from test_demo.temporal.workflows import (
    TASK_QUEUE,
    DeepxOrchestratorInput,
    DeepxOrchestratorWorkflow,
)


async def connect_temporal_client() -> Client:
    load_dotenv()
    return await Client.connect(
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


async def start_orchestrator_workflow(
    prompt: str,
    session_id: str,
    *,
    workflow_id: str | None = None,
) -> tuple[WorkflowHandle[Any, Any], str, str]:
    """Return ``(handle, workflow_id, session_id)``."""
    client = await connect_temporal_client()
    wf_id = workflow_id or f"deepx-{uuid.uuid4().hex[:12]}"
    inp = DeepxOrchestratorInput(prompt=prompt, session_id=session_id)
    handle = await client.start_workflow(
        DeepxOrchestratorWorkflow.run,
        inp,
        id=wf_id,
        task_queue=TASK_QUEUE,
    )
    return handle, wf_id, session_id
