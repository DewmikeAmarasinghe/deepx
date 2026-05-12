"""Public entry points for launching a deepx agent REPL session."""

from __future__ import annotations

import asyncio

from rich.console import Console

from deepx.factory import DeepAgentRunner, DeepRunBinding

from deepx_cli._stream import run_stream_until_settled
from deepx_cli.repl import parse_session_arg, run_interactive_repl


# ---------------------------------------------------------------------------
# Turn implementations
# ---------------------------------------------------------------------------


async def _streaming_turn(
    binding: DeepRunBinding, user_input: str, console: Console
) -> None:
    await run_stream_until_settled(binding, user_input, console, stream_text=True)


async def _sync_turn(
    binding: DeepRunBinding, user_input: str, console: Console
) -> None:
    await run_stream_until_settled(binding, user_input, console, stream_text=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_chat_stream(
    runner: DeepAgentRunner,
    *,
    session_id: str | None = None,
) -> None:
    """Start an interactive REPL with token streaming.

    Args:
        runner:     The agent runner to use.
        session_id: Explicit session id. Falls back to ``--session`` argv flag
                    when omitted.
    """
    sid = parse_session_arg() if session_id is None else session_id
    asyncio.run(run_interactive_repl(runner, session_id=sid, run_turn=_streaming_turn))


def run_chat_sync(
    runner: DeepAgentRunner,
    *,
    session_id: str | None = None,
) -> None:
    """Start an interactive REPL without token streaming.

    Args:
        runner:     The agent runner to use.
        session_id: Explicit session id. Falls back to ``--session`` argv flag
                    when omitted.
    """
    sid = parse_session_arg() if session_id is None else session_id
    asyncio.run(run_interactive_repl(runner, session_id=sid, run_turn=_sync_turn))


__all__ = ["run_chat_stream", "run_chat_sync"]