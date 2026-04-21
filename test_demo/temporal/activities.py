"""Temporal activities for the Deepx demo.

Model calls are still executed via ``OpenAIAgentsPlugin`` (registered on the Temporal client).
This module hosts **application** work that must not run inside workflow sandbox tasks—e.g. a full
``DeepAgentRunner.run`` with SQLite-backed :class:`agents.memory.SQLiteSession` (blocking I/O and
thread pools).
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from temporalio import activity

from deepx_cli.chat_stream import run_binding_until_settled


@dataclass
class RunOrchestratorActivityInput:
    prompt: str
    session_id: str
    resume: bool = False


@activity.defn
async def run_orchestrator_activity(inp: RunOrchestratorActivityInput) -> str:
    from test_demo import orchestrator as orch

    runner = orch.orchestrator_runner
    binding = runner.bind(inp.session_id, resume=inp.resume)
    result = await run_binding_until_settled(
        binding, inp.prompt, console=Console(highlight=False)
    )
    return str(result.final_output)
