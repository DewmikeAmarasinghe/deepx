"""Shared argparse entry for running a :class:`~deepx.factory.DeepAgentRunner` in the REPL."""

from __future__ import annotations

import argparse

from deepx.factory import DeepAgentRunner


def run_interactive_cli(
    runner: DeepAgentRunner | None,
    *,
    description: str,
    unavailable: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Resume this session id (must be used with --chat or --chat_sync).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn session with streaming (default if --chat_sync not set).",
    )
    parser.add_argument(
        "--chat_sync",
        action="store_true",
        help="Interactive multi-turn session without token streaming.",
    )
    args, _rest = parser.parse_known_args()

    if runner is None:
        parser.error(
            unavailable or "This agent is not available in the current configuration."
        )

    if args.session is not None and not args.chat and not args.chat_sync:
        parser.error("--session requires --chat or --chat_sync")

    from deepx_cli.chat_stream import run_chat_stream
    from deepx_cli.chat_sync import run_chat_sync

    sid = args.session
    if not args.chat_sync:
        run_chat_stream(runner, session_id=sid)
    else:
        run_chat_sync(runner, session_id=sid)


__all__ = ["run_interactive_cli"]
