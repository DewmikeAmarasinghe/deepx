from __future__ import annotations

import asyncio
import os
import uuid

from rich.console import Console

from deepx.factory import DeepAgentRunner
from deepx_cli.chat_stream import run_stream_until_settled
from deepx_cli.temporal_run import run_via_temporal


def use_temporal() -> bool:
    """True when runs should go through Temporal (demo workflow + activity).

    ``USE_TEMPORAL``.
    """
    raw = (
        (os.environ.get("USE_TEMPORAL") or os.environ.get("DEEPX_USE_TEMPORAL") or "")
        .strip()
        .lower()
    )
    return raw in ("1", "true", "yes", "on")


async def _chat_loop_async(
    runner: DeepAgentRunner, *, user_name: str, resume_hint: str | None
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    agent_label = runner._agent_name.replace("_", " ").title()

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    turn = 0
    ut = use_temporal()
    if ut:
        console.print(
            "[dim]USE_TEMPORAL: runs go through Temporal (no token streaming from the CLI). "
            "Interactive tool approval is not available in this mode—avoid tools that set "
            "needs_approval, or run without USE_TEMPORAL.[/dim]\n"
        )

    try:
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

            if ut:
                out = await run_via_temporal(
                    prompt=user_input,
                    session_id=sid,
                    resume=(turn > 0),
                )
                console.print()
                console.print(f"[bold]{agent_label}:[/bold] {out}\n")
            else:
                binding = runner.bind(sid, resume=(turn > 0))
                stream = await run_stream_until_settled(binding, user_input, console)
                console.print()
                console.print(f"[bold]{agent_label}:[/bold] {stream.final_output}\n")
            turn += 1
    finally:
        pass


def run_chat(
    runner: DeepAgentRunner,
    *,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Interactive multi-turn chat with SDK approvals (terminal) and optional streaming."""
    asyncio.run(_chat_loop_async(runner, user_name=user_name, resume_hint=resume_hint))


def run_once(
    runner: DeepAgentRunner,
    task: str,
    *,
    session_id: str | None = None,
) -> None:
    """Single task, non-interactive except approval prompts."""
    console = Console(highlight=False)
    sid = session_id or os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    ut = use_temporal()

    async def _once() -> None:
        if ut:
            out = await run_via_temporal(prompt=task, session_id=sid, resume=False)
        else:
            binding = runner.bind(sid, resume=False)
            stream = await run_stream_until_settled(binding, task, console)
            out = stream.final_output
        console.print("\n" + "=" * 70)
        console.print(out)
        console.print("=" * 70)
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    asyncio.run(_once())


def run_interactive(
    runner: DeepAgentRunner,
    *,
    session_id: str | None = None,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Backward-compatible alias for :func:`run_chat`."""
    _ = session_id
    run_chat(runner, user_name=user_name, resume_hint=resume_hint)


__all__ = [
    "run_chat",
    "run_interactive",
    "run_once",
    "use_temporal",
]
