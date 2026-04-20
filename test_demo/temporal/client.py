"""Temporal client helpers for the orchestrator demo."""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin

from test_demo.temporal.workflows import (
    TASK_QUEUE,
    DeepxOrchestratorInput,
    DeepxOrchestratorWorkflow,
)


def _temporal_address() -> str:
    return (os.environ.get("TEMPORAL_ADDRESS") or "localhost:7233").strip()


def _temporal_namespace() -> str:
    return (os.environ.get("TEMPORAL_NAMESPACE") or "default").strip()


def openai_agents_plugin() -> OpenAIAgentsPlugin:
    """Shared plugin for client and worker (model activity timeouts, payload converter, tracing)."""
    return OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(minutes=30),
        ),
    )


async def connect_temporal_client() -> Client:
    load_dotenv()
    plugin = openai_agents_plugin()
    return await Client.connect(
        _temporal_address(),
        namespace=_temporal_namespace(),
        plugins=[plugin],
    )


async def run_orchestrator_workflow_and_wait(
    prompt: str,
    session_id: str,
    *,
    resume: bool = False,
    workflow_id: str | None = None,
) -> str:
    """Start the demo workflow and return the orchestrator's final text output."""
    client = await connect_temporal_client()
    wf_id = workflow_id or f"deepx-{session_id}-{uuid.uuid4().hex[:10]}"
    inp = DeepxOrchestratorInput(prompt=prompt, session_id=session_id, resume=resume)
    handle = await client.start_workflow(
        DeepxOrchestratorWorkflow.run,
        inp,
        id=wf_id,
        task_queue=TASK_QUEUE,
    )
    return await handle.result()


async def start_orchestrator_workflow(
    prompt: str,
    session_id: str,
    *,
    resume: bool = False,
    workflow_id: str | None = None,
) -> tuple[WorkflowHandle[Any, Any], str, str]:
    """Return ``(handle, workflow_id, session_id)``."""
    client = await connect_temporal_client()
    wf_id = workflow_id or f"deepx-{uuid.uuid4().hex[:12]}"
    inp = DeepxOrchestratorInput(prompt=prompt, session_id=session_id, resume=resume)
    handle = await client.start_workflow(
        DeepxOrchestratorWorkflow.run,
        inp,
        id=wf_id,
        task_queue=TASK_QUEUE,
    )
    return handle, wf_id, session_id
