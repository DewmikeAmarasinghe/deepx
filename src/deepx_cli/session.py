from __future__ import annotations

import asyncio
import os
import uuid

from rich.console import Console

from deepx.factory import DeepAgentRunner
from deepx_cli.chat_stream import run_stream_until_settled
from deepx_cli.temporal_run import run_via_temporal


def use_temporal() -> bool:
    """True when runs execute via Temporal workflow (``DeepxOrchestratorWorkflow``).

    ``USE_TEMPORAL``.
    """
    raw = (os.environ.get("USE_TEMPORAL") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _display_agent_name(agent_name: str) -> str:
    return agent_name.replace("_", " ").title()


async def _chat_loop_async(
    runner: DeepAgentRunner, *, user_name: str, resume_hint: str | None
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    default_label = _display_agent_name(runner._agent_name)

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    turn = 0
    ut = use_temporal()
    if ut:
        console.print(
            "[dim]USE_TEMPORAL: worker runs the workflow (no token streaming in this CLI). "
            "Tool approvals use this terminal + Temporal signals.[/dim]\n"
        )

    chat_handle = None
    chat_seq = -1
    hitl_pump: asyncio.Task[None] | None = None
    temporal_client = None

    try:
        if ut:
            from test_demo.temporal.client import (
                connect_temporal_client,
                ensure_chat_workflow,
                end_temporal_chat_session,
                run_temporal_turn_chat,
                temporal_hitl_pump,
            )

            temporal_client = await connect_temporal_client()
            chat_handle = await ensure_chat_workflow(temporal_client, sid)
            hitl_pump = asyncio.create_task(temporal_hitl_pump(chat_handle, console))

        while True:
            try:
                user_input = input(f"{user_name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input == "/bye":
                if ut and chat_handle is not None:
                    await end_temporal_chat_session(chat_handle)
                console.print("Goodbye.")
                break

            if ut:
                assert chat_handle is not None
                out, chat_seq = await run_temporal_turn_chat(
                    chat_handle, user_input, chat_seq
                )
                console.print(f"[bold]{default_label}:[/bold] {out}\n")
            else:
                binding = runner.bind(sid, resume=(turn > 0))
                console.print()
                stream = await run_stream_until_settled(
                    binding, user_input, console, stream_text=True
                )
                try:
                    stream_title = _display_agent_name(stream.last_agent.name)
                except Exception:
                    stream_title = default_label
                console.print(f"[dim]— {stream_title}[/dim]\n")
            turn += 1
    finally:
        if hitl_pump is not None:
            hitl_pump.cancel()
            try:
                await hitl_pump
            except asyncio.CancelledError:
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
        console.print(f"[bold]You:[/bold] {task}\n")
        if ut:
            out = await run_via_temporal(
                prompt=task, session_id=sid, resume=False, console=console
            )
            body = out
            title = _display_agent_name(runner._agent_name)
            console.print(f"[bold]{title}:[/bold] {body}\n")
        else:
            binding = runner.bind(sid, resume=False)
            default_t = _display_agent_name(runner._agent_name)
            console.print()
            stream = await run_stream_until_settled(
                binding, task, console, stream_text=True
            )
            try:
                stream_title = _display_agent_name(stream.last_agent.name)
            except Exception:
                stream_title = default_t
            console.print(f"[dim]— {stream_title}[/dim]\n")
        console.print(f"[dim]Session:[/dim] {sid}\n")

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
