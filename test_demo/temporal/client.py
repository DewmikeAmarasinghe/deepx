"""Temporal client helpers for the orchestrator demo."""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from temporalio.client import Client, WorkflowHandle
from temporalio.common import RetryPolicy
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.service import RPCError, RPCStatusCode

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
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                backoff_coefficient=2.0,
                maximum_attempts=5,
            ),
        ),
        add_temporal_spans=True,
    )


async def connect_temporal_client() -> Client:
    load_dotenv()
    plugin = openai_agents_plugin()
    return await Client.connect(
        _temporal_address(),
        namespace=_temporal_namespace(),
        plugins=[plugin],
    )


def chat_workflow_id(session_id: str) -> str:
    return f"deepx-chat-{session_id}"


async def temporal_hitl_pump(
    handle: WorkflowHandle[Any, Any],
    console: Console,
) -> None:
    from deepx_cli.approvals import approval_choice_from_meta

    try:
        while True:
            await asyncio.sleep(0.12)
            pending = await handle.query(DeepxOrchestratorWorkflow.hitl_pending)
            if not pending:
                continue
            choice = await asyncio.to_thread(
                approval_choice_from_meta, console, pending
            )
            await handle.signal(DeepxOrchestratorWorkflow.hitl_approval, choice)
    except asyncio.CancelledError:
        return


async def ensure_chat_workflow(
    client: Client,
    session_id: str,
) -> WorkflowHandle[Any, Any]:
    wf_id = chat_workflow_id(session_id)
    inp = DeepxOrchestratorInput(session_id=session_id, multi_turn=True)
    try:
        return await client.start_workflow(
            DeepxOrchestratorWorkflow.run,
            inp,
            id=wf_id,
            task_queue=TASK_QUEUE,
        )
    except RPCError as e:
        if e.status != RPCStatusCode.ALREADY_EXISTS:
            raise
        return client.get_workflow_handle(wf_id)


async def run_temporal_turn_chat(
    handle: WorkflowHandle[Any, Any],
    text: str,
    last_seen_seq: int,
) -> tuple[str, int]:
    await handle.signal(DeepxOrchestratorWorkflow.user_message, text)
    while True:
        await asyncio.sleep(0.05)
        st = await handle.query(DeepxOrchestratorWorkflow.get_chat_status)
        if st["seq"] > last_seen_seq:
            return str(st["last_output"]), int(st["seq"])


async def end_temporal_chat_session(handle: WorkflowHandle[Any, Any]) -> None:
    await handle.signal(DeepxOrchestratorWorkflow.end_session)


async def run_orchestrator_workflow_and_wait(
    prompt: str,
    session_id: str,
    *,
    resume: bool = False,
    workflow_id: str | None = None,
    console: Console | None = None,
) -> str:
    """Start the demo workflow and return the orchestrator's final text output."""
    client = await connect_temporal_client()
    wf_id = workflow_id or f"deepx-{session_id}-{uuid.uuid4().hex[:10]}"
    inp = DeepxOrchestratorInput(
        session_id=session_id,
        prompt=prompt,
        resume=resume,
        multi_turn=False,
    )
    handle = await client.start_workflow(
        DeepxOrchestratorWorkflow.run,
        inp,
        id=wf_id,
        task_queue=TASK_QUEUE,
    )
    pump: asyncio.Task[None] | None = None
    if console is not None:
        pump = asyncio.create_task(temporal_hitl_pump(handle, console))
    try:
        return await handle.result()
    finally:
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
