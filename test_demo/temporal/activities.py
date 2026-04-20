"""Temporal activities for the Deepx demo.

Model calls are still executed via ``OpenAIAgentsPlugin`` (registered on the Temporal client).
This module hosts **application** work that must not run inside workflow sandbox tasks—e.g. a full
``DeepAgentRunner.run`` with SQLite-backed :class:`agents.memory.SQLiteSession` (blocking I/O and
thread pools).
"""

from __future__ import annotations

from dataclasses import dataclass

from temporalio import activity


@dataclass
class RunOrchestratorActivityInput:
    prompt: str
    session_id: str
    resume: bool = False


@activity.defn
async def run_orchestrator_activity(inp: RunOrchestratorActivityInput) -> str:
    from test_demo import orchestrator as orch

    runner = orch.build_orchestrator_runner()
    result = await runner.run(
        inp.prompt,
        session_id=inp.session_id,
        resume=inp.resume,
    )
    return str(result.output)
