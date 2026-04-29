from __future__ import annotations

import asyncio

from rich.console import Console

from deepx.factory import DeepAgentRunner, DeepRunBinding
from deepx_cli.chat_stream import run_stream_until_settled
from deepx_cli.session import parse_cli_session_arg, run_interactive_repl


async def _sync_turn(
    binding: DeepRunBinding, user_input: str, console: Console
) -> None:
    """Same nested tool visibility as streaming, but no token-by-token model deltas."""
    await run_stream_until_settled(binding, user_input, console, stream_text=False)


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
