from __future__ import annotations

import asyncio

from rich.console import Console

from deepx.factory import DeepAgentRunner, DeepRunBinding
from deepx_cli.session import parse_cli_session_arg, run_interactive_repl


async def _sync_turn(
    binding: DeepRunBinding, user_input: str, console: Console
) -> None:
    result = await binding.run(user_input)
    console.print(str(result.final_output or ""))


def run_chat_sync(
    runner: DeepAgentRunner,
    *,
    session_id: str | None = None,
) -> None:
    """Interactive loop without token streaming. Uses ``--session`` from argv when ``session_id`` is omitted."""
    sid = parse_cli_session_arg() if session_id is None else session_id
    asyncio.run(
        run_interactive_repl(
            runner,
            session_id=sid,
            run_turn=_sync_turn,
        )
    )


__all__ = ["run_chat_sync"]
