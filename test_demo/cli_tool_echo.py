"""Demo CLI: print each tool call (agent, tool name, full arguments) without tool output."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

from agents.result import RunResultStreaming
from agents.run_state import RunState
from rich.console import Console
from rich.panel import Panel

from deepx.factory import DeepAgentRunner, DeepRunBinding
from deepx_cli.approvals import apply_choices_to_state
from deepx_cli.temporal_run import run_via_temporal


def _panel_title(agent_name: str) -> str:
    return agent_name.replace("_", " ").title()


def use_temporal() -> bool:
    raw = (os.environ.get("USE_TEMPORAL") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


async def _drain_tool_calls_only(stream: RunResultStreaming, console: Console) -> None:
    async for event in stream.stream_events():
        if event.type != "run_item_stream_event":
            continue
        item = event.item
        name = getattr(item, "type", None) or ""
        if name != "tool_call_item":
            continue
        raw = getattr(item, "raw_item", None)
        agent_obj = getattr(item, "agent", None)
        ag_name = getattr(agent_obj, "name", None) if agent_obj is not None else ""
        if isinstance(raw, dict):
            tname = str(raw.get("name", "") or "")
            args_raw = raw.get("arguments", "")
        else:
            tname = str(getattr(raw, "name", "") or "")
            args_raw = getattr(raw, "arguments", "")
        if isinstance(args_raw, (dict, list)):
            args_str = json.dumps(args_raw, ensure_ascii=False)
        else:
            args_str = str(args_raw) if args_raw is not None else ""
        console.print(f"[cyan]{ag_name}[/cyan]  [bold]{tname}[/bold]  {args_str}")


async def run_stream_until_settled_with_tool_echo(
    binding: DeepRunBinding,
    inp: str | RunState,
    console: Console,
) -> RunResultStreaming:
    stream = binding.run_streamed(inp)
    await _drain_tool_calls_only(stream, console)
    while stream.interruptions:
        console.print()
        state = stream.to_state()
        apply_choices_to_state(state, list(stream.interruptions), console)
        stream = binding.run_streamed(state)
        await _drain_tool_calls_only(stream, console)
    return stream


async def _chat_loop_async(
    runner: DeepAgentRunner, *, user_name: str, resume_hint: str | None
) -> None:
    console = Console(highlight=False)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    default_label = _panel_title(runner._agent_name)
    w = console.size.width or 120

    if resume_hint:
        console.print(f"\n[dim]Session:[/dim] {sid}  — resume: `{resume_hint} {sid}`\n")
    else:
        console.print(f"\n[dim]Session:[/dim] {sid}\n")

    turn = 0
    if use_temporal():
        console.print(
            "[dim]USE_TEMPORAL: worker runs the workflow (no token streaming). "
            "Tool approvals use the worker stdin when interactive.[/dim]\n"
        )

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

        if use_temporal():
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
            stream = await run_stream_until_settled_with_tool_echo(
                binding, user_input, console
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


def run_chat(
    runner: DeepAgentRunner,
    *,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    asyncio.run(_chat_loop_async(runner, user_name=user_name, resume_hint=resume_hint))


def run_once(
    runner: DeepAgentRunner,
    task: str,
    *,
    session_id: str | None = None,
) -> None:
    console = Console(highlight=False)
    sid = session_id or os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    w = console.size.width or 120

    async def _once() -> None:
        if use_temporal():
            out = await run_via_temporal(prompt=task, session_id=sid, resume=False)
            body = out
            title = _panel_title(runner._agent_name)
        else:
            binding = runner.bind(sid, resume=False)
            stream = await run_stream_until_settled_with_tool_echo(
                binding, task, console
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


__all__ = [
    "run_chat",
    "run_once",
    "run_stream_until_settled_with_tool_echo",
    "use_temporal",
]
