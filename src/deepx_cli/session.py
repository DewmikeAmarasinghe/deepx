"""Shared interactive REPL for ``deepx_cli``.

Requires the ``demo`` extra (``rich``, ``prompt_toolkit``) — the core ``deepx`` package does not
depend on them.
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from deepx.factory import DeepAgentRunner, DeepRunBinding
from deepx_cli.hitl import create_terminal_hitl

_INPUT_HELP = (
    "[dim]Press Enter to send. /bye to quit. "
    "(Paste can include newlines; for a long draft, paste as one message.)[/dim]"
)

_RESUME_SESSION = "python -m test_demo.orchestrator --chat/--chat_sync --session"


def _print_resume_hint(console: Console, sid: str) -> None:
    console.print(
        f"[dim]To resume next time run:[/dim] {_RESUME_SESSION} {sid}  "
    )

def _display_agent_name(agent_name: str) -> str:
    return agent_name.replace("_", " ").title()


def _resolve_session(cli_session: str | None) -> tuple[str, bool]:
    """New id unless ``cli_session`` is set (resume). No environment fallback."""
    if cli_session:
        return cli_session.strip(), True
    return uuid.uuid4().hex[:12], False


def _user_prompt_session() -> PromptSession:
    """Single-line primary mode: Enter accepts (see prompt_toolkit \"Multiline input\" docs)."""
    return PromptSession(history=InMemoryHistory(), multiline=False)


async def _read_turn(session: PromptSession) -> str:
    return (await session.prompt_async("  ", default="")).strip()


def parse_cli_session_arg() -> str | None:
    """Read ``--session`` from argv (``parse_known_args``)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session", default=None)
    args, _ = parser.parse_known_args()
    return args.session


async def run_interactive_repl(
    runner: DeepAgentRunner,
    *,
    session_id: str | None,
    run_turn: Callable[[DeepRunBinding, str, Console], Awaitable[None]],
) -> None:
    console = Console(highlight=False)
    sid, resuming = _resolve_session(session_id)
    hitl = create_terminal_hitl(console)
    pt_session = _user_prompt_session()

    if resuming:
        console.print(f"\n[dim]Resuming session:[/dim] {sid}\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}")
        _print_resume_hint(console, sid)
        console.print()

    console.print(_INPUT_HELP)
    console.print()

    turn = 0
    while True:
        try:
            console.print("[bold]You:[/bold]")
            user_input = await _read_turn(pt_session)
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n[dim]Exiting.[/dim]")
            break

        if not user_input:
            continue
        if user_input.strip() == "/bye":
            console.print()
            console.print("[dim]Goodbye.[/dim]")
            console.print()
            _print_resume_hint(console, sid)
            console.print()
            break

        binding = runner.bind(sid, resume=(resuming or turn > 0), hitl=hitl)
        console.print()
        console.print()
        console.print(f"[bold]{_display_agent_name(runner._agent_name)}:[/bold]")
        await run_turn(binding, user_input, console)
        console.print()
        console.print()
        turn += 1


__all__ = [
    "parse_cli_session_arg",
    "run_interactive_repl",
]
