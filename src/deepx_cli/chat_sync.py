from __future__ import annotations

from typing import Any

from agents.result import RunResult
from agents.run_state import RunState
from rich.console import Console

from deepx.factory import DeepRunBinding


async def run_turn_sync(
    binding: DeepRunBinding, inp: str | RunState[Any, Any], console: Console
) -> RunResult:
    """One non-streamed model turn (Deepx HITL still runs inside gated tool invokes)."""
    _ = console
    return await binding.run(inp)


__all__ = ["run_turn_sync"]
