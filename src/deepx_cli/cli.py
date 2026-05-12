"""Unified argparse entry point for running a DeepAgentRunner in the REPL."""

from __future__ import annotations

import argparse

from deepx.factory import DeepAgentRunner


def run_interactive_cli(
    runner: DeepAgentRunner | None,
    *,
    description: str,
    unavailable: str | None = None,
) -> None:
    """Parse CLI flags and launch an interactive REPL session.

    Args:
        runner:      The agent runner to use. Pass ``None`` when unavailable
                     (e.g. missing API key) — the parser will exit with
                     *unavailable* as the error message.
        description: ``argparse`` description shown in ``--help``.
        unavailable: Error text emitted when *runner* is ``None``.

    Flags
    -----
    --chat        Streaming token-by-token REPL (default).
    --chat_sync   Non-streaming REPL (tool calls still visible).
    --session ID  Resume a prior session (requires ``--chat`` or ``--chat_sync``).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Resume this session id (requires --chat or --chat_sync).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn session with streaming (default).",
    )
    parser.add_argument(
        "--chat_sync",
        action="store_true",
        help="Interactive multi-turn session without token streaming.",
    )
    args, _ = parser.parse_known_args()

    if runner is None:
        parser.error(
            unavailable or "This agent is not available in the current configuration."
        )

    if args.session is not None and not args.chat and not args.chat_sync:
        parser.error("--session requires --chat or --chat_sync.")

    from deepx_cli.run import run_chat_stream, run_chat_sync

    if args.chat_sync:
        run_chat_sync(runner, session_id=args.session)
    else:
        run_chat_stream(runner, session_id=args.session)


__all__ = ["run_interactive_cli"]