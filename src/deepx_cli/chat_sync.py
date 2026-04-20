from __future__ import annotations

from typing import Any

from agents.result import RunResult
from agents.run_state import RunState
from rich.console import Console

from deepx.factory import DeepRunBinding

from deepx_cli.approvals import apply_choices_to_state


async def run_until_settled(
    binding: DeepRunBinding, inp: str | RunState[Any, Any], console: Console
) -> RunResult:
    result = await binding.run(inp)
    while result.interruptions:
        console.print()
        state = result.to_state()
        apply_choices_to_state(state, result.interruptions, console)
        result = await binding.run(state)
    return result


__all__ = ["run_until_settled"]
