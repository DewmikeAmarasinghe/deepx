from __future__ import annotations

import asyncio
import os
import uuid

from rich.console import Console
from rich.panel import Panel

from deepx.factory import DeepAgentRunner
from deepx_cli.chat_stream import run_stream_until_settled
from deepx_cli.temporal_run import run_via_temporal


def use_temporal() -> bool:
    """True when runs execute via Temporal workflow (``DeepxOrchestratorWorkflow``).

    ``USE_TEMPORAL``.
    """
    raw = (os.environ.get("USE_TEMPORAL") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _panel_title(agent_name: str) -> str:
    return agent_name.replace("_", " ").title()


async def _chat_loop_async(
    runner: DeepAgentRunner, *, user_name: str, resume_hint: str | None
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    default_label = _panel_title(runner._agent_name)

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    turn = 0
    ut = use_temporal()
    if ut:
        console.print(
            "[dim]USE_TEMPORAL: worker runs the workflow (no token streaming in this CLI). "
            "Tool approvals use the worker stdin when interactive.[/dim]\n"
        )

    w = console.size.width or 120
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

            console.print()
            console.print(
                Panel(
                    user_input,
                    title=user_name,
                    border_style="blue",
                    expand=True,
                    width=w,
                )
            )
            console.print()

            if ut:
                out = await run_via_temporal(
                    prompt=user_input,
                    session_id=sid,
                    resume=(turn > 0),
                )
                console.print(
                    Panel(
                        out,
                        title=default_label,
                        border_style="green",
                        expand=True,
                        width=w,
                    )
                )
                console.print()
            else:
                binding = runner.bind(sid, resume=(turn > 0))
                stream = await run_stream_until_settled(
                    binding, user_input, console, stream_text=True
                )
                try:
                    active = stream.last_agent.name
                except Exception:
                    active = runner._agent_name
                title = _panel_title(active)
                console.print(
                    Panel(
                        str(stream.final_output),
                        title=title,
                        border_style="green",
                        expand=True,
                        width=w,
                    )
                )
                console.print()
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

    w = console.size.width or 120

    async def _once() -> None:
        if ut:
            out = await run_via_temporal(prompt=task, session_id=sid, resume=False)
            body = out
            title = _panel_title(runner._agent_name)
        else:
            binding = runner.bind(sid, resume=False)
            stream = await run_stream_until_settled(
                binding, task, console, stream_text=True
            )
            body = str(stream.final_output)
            try:
                title = _panel_title(stream.last_agent.name)
            except Exception:
                title = _panel_title(runner._agent_name)
        console.print(
            Panel(body, title=title, border_style="green", expand=True, width=w)
        )
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
