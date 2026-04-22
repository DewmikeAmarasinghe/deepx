from __future__ import annotations

import asyncio
import os
import uuid

from rich.console import Console

from deepx.factory import DeepAgentRunner
from deepx_cli.chat_stream import run_stream_until_settled
from deepx_cli.chat_sync import run_turn_sync
from deepx_cli.hitl import create_terminal_hitl


def _display_agent_name(agent_name: str) -> str:
    return agent_name.replace("_", " ").title()


async def _chat_loop_stream_async(
    runner: DeepAgentRunner, *, user_name: str, resume_hint: str | None
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    hitl = create_terminal_hitl(console)

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    turn = 0
    while True:
        try:
            user_input = input(f"{user_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == "/bye":
            console.print("Goodbye.")
            break

        binding = runner.bind(sid, resume=(turn > 0), hitl=hitl)
        console.print(f"\n[bold]{_display_agent_name(runner._agent_name)}:[/bold]\n")
        await run_stream_until_settled(
            binding, user_input, console, stream_text=True
        )
        console.print()
        turn += 1


def run_chat_stream(
    runner: DeepAgentRunner,
    *,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Interactive multi-turn chat with streaming and Deepx HITL."""
    asyncio.run(
        _chat_loop_stream_async(runner, user_name=user_name, resume_hint=resume_hint)
    )


async def _chat_loop_sync_async(
    runner: DeepAgentRunner, *, user_name: str, resume_hint: str | None
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    hitl = create_terminal_hitl(console)

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    turn = 0
    while True:
        try:
            user_input = input(f"{user_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == "/bye":
            console.print("Goodbye.")
            break

        binding = runner.bind(sid, resume=(turn > 0), hitl=hitl)
        console.print(f"\n[bold]{_display_agent_name(runner._agent_name)}:[/bold]\n")
        result = await run_turn_sync(binding, user_input, console)
        console.print(str(result.final_output or ""))
        console.print()
        turn += 1


def run_chat_sync(
    runner: DeepAgentRunner,
    *,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Interactive multi-turn chat without token streaming (one ``Runner.run`` per message)."""
    asyncio.run(
        _chat_loop_sync_async(runner, user_name=user_name, resume_hint=resume_hint)
    )


def run_once(
    runner: DeepAgentRunner,
    task: str,
    *,
    session_id: str | None = None,
) -> None:
    """Single task with streaming; gated tools prompt on this terminal."""
    console = Console(highlight=False)
    sid = session_id or os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    hitl = create_terminal_hitl(console)

    async def _once() -> None:
        console.print(f"[bold]You:[/bold] {task}\n")
        binding = runner.bind(sid, resume=False, hitl=hitl)
        console.print(f"[bold]{_display_agent_name(runner._agent_name)}:[/bold]\n")
        await run_stream_until_settled(binding, task, console, stream_text=True)
        console.print()
        console.print(f"[dim]Session:[/dim] {sid}\n")

    asyncio.run(_once())


__all__ = [
    "run_chat_stream",
    "run_chat_sync",
    "run_once",
]
