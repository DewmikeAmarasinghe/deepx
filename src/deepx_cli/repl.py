"""Interactive REPL loop shared by all deepx_cli run modes.

Requires the ``demo`` extra (``rich``, ``prompt_toolkit``).
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import EmacsInsertMode, ViInsertMode
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from rich.console import Console

from deepx.factory import DeepAgentRunner, DeepRunBinding
from deepx_cli.hitl import create_terminal_hitl

_HELP_LINES = [
    "[dim]Press Enter to send[/dim]",
    "[dim]Esc+Enter or Alt+Enter for a new line[/dim]",
    "[dim]/bye to quit[/dim]",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_session_arg() -> str | None:
    """Read ``--session <ID>`` from argv without consuming other flags."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session", default=None)
    args, _ = parser.parse_known_args()
    return args.session


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _resume_hint(runner: DeepAgentRunner, sid: str) -> str:
    name = getattr(runner, "_agent_name", None) or "agent"
    agent_file = f"test_demo/{name}.py"
    return f"[dim]Resume:  python {agent_file} --chat --session {sid}[/dim]"


def _make_keybindings() -> KeyBindings:
    """Bind Esc+Enter and Alt+Enter to insert a newline for multiline input."""
    kb = KeyBindings()

    @kb.add("escape", "enter", filter=EmacsInsertMode() | ViInsertMode())
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return kb


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


async def run_interactive_repl(
    runner: DeepAgentRunner,
    *,
    session_id: str | None,
    run_turn: Callable[[DeepRunBinding, str, Console], Awaitable[None]],
) -> None:
    """Run the interactive REPL until the user exits.

    Spacing contract
    ----------------
    One blank line separates every distinct region:

        <blank>
        Session: <id>
        Press Enter to send
        Esc+Enter or Alt+Enter for a new line
        /bye to quit
        <blank>
        you:
          <input>
        <blank>
        <turn output>
        <blank>
        you:
          ...

    Args:
        runner:     The agent runner to use.
        session_id: Explicit session id to resume; ``None`` generates a new one.
        run_turn:   Async callable that executes one user turn.
    """
    console = Console(highlight=False)
    resuming = bool(session_id)
    sid = session_id or _new_session_id()
    hitl = create_terminal_hitl(console)
    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        multiline=False,
        key_bindings=_make_keybindings(),
    )

    # -- Session banner ------------------------------------------------------
    console.print()
    label = "Resuming session" if resuming else "Session"
    console.print(f"[dim]{label}:[/dim] {sid}")
    for line in _HELP_LINES:
        console.print(line)
    console.print()

    turn = 0
    while True:
        # -- Prompt ----------------------------------------------------------
        try:
            console.print("[bold]you:[/bold]")
            user_input: str = (
                await pt_session.prompt_async("  ", default="")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print(_resume_hint(runner, sid))
            console.print()
            break

        if not user_input:
            continue
        if user_input == "/bye":
            console.print()
            console.print("[dim]Goodbye.[/dim]")
            console.print()
            console.print(_resume_hint(runner, sid))
            console.print()
            break

        # -- Turn ------------------------------------------------------------
        # New user line: fresh AgentContext; resume=True only when user passed --session
        # (reloads saved plan from logs when debug wrote it).
        binding = runner.bind(sid, resume=resuming, hitl=hitl)
        console.print()
        await run_turn(binding, user_input, console)
        console.print()
        turn += 1


__all__ = ["parse_session_arg", "run_interactive_repl"]