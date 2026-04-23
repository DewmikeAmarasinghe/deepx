from __future__ import annotations

import asyncio
import os
import uuid

from rich.console import Console

from deepx.factory import DeepAgentRunner
from deepx_cli.chat_stream import run_stream_until_settled
from deepx_cli.hitl import create_terminal_hitl
from deepx_cli.input_multiline import read_user_turn


def _display_agent_name(agent_name: str) -> str:
    return agent_name.replace("_", " ").title()


async def _chat_loop_stream_async(
    runner: DeepAgentRunner,
    *,
    user_name: str,
    resume_hint: str | None,
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    hitl = create_terminal_hitl(console)

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")
    console.print(
        '[dim]Multi-line: first line exactly """ then paste; end with """ and Enter.[/dim]\n'
    )

    turn = 0
    while True:
        try:
            console.print("[bold]You:[/bold]")
            user_input = read_user_turn()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == "/bye":
            console.print("Goodbye.")
            break

        binding = runner.bind(sid, resume=(turn > 0), hitl=hitl)
        console.print(f"\n[bold]{_display_agent_name(runner._agent_name)}:[/bold]")
        await run_stream_until_settled(
            binding,
            user_input,
            console,
            stream_text=True,
        )
        console.print("\n\n")
        turn += 1


def run_chat_stream(
    runner: DeepAgentRunner,
    *,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Interactive multi-turn chat with streaming and Deepx HITL."""
    _ = user_name
    asyncio.run(
        _chat_loop_stream_async(
            runner,
            user_name=user_name,
            resume_hint=resume_hint,
        )
    )


async def _chat_loop_sync_async(
    runner: DeepAgentRunner,
    *,
    user_name: str,
    resume_hint: str | None,
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    hitl = create_terminal_hitl(console)

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")
    console.print(
        '[dim]Multi-line: first line exactly """ then paste; end with """ and Enter.[/dim]\n'
    )

    turn = 0
    while True:
        try:
            console.print("[bold]You:[/bold]")
            user_input = read_user_turn()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == "/bye":
            console.print("Goodbye.")
            break

        binding = runner.bind(sid, resume=(turn > 0), hitl=hitl)
        console.print(f"\n[bold]{_display_agent_name(runner._agent_name)}:[/bold]")
        result = await binding.run(user_input)
        console.print(str(result.final_output or ""))
        console.print("\n\n")
        turn += 1


def run_chat_sync(
    runner: DeepAgentRunner,
    *,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Interactive multi-turn chat without token streaming (one ``Runner.run`` per message)."""
    _ = user_name
    asyncio.run(
        _chat_loop_sync_async(runner, user_name=user_name, resume_hint=resume_hint)
    )


__all__ = [
    "run_chat_stream",
    "run_chat_sync",
]
