from __future__ import annotations

import asyncio
import sys
from typing import Any

from agents.items import ToolCallItem, ToolCallOutputItem
from agents.result import RunResultStreaming
from agents.run_state import RunState
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from rich.console import Console

from deepx.factory import DeepAgentRunner, DeepRunBinding
from deepx_cli.session import parse_cli_session_arg, run_interactive_repl


def _write_stream_status(msg: str) -> None:
    if len(msg) > 120:
        msg = msg[:117] + "..."
    sys.stderr.write(f"\r{msg}\033[K")
    sys.stderr.flush()


def _finish_stream_status() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()


def _format_run_item_status(event: RunItemStreamEvent) -> str:
    item = event.item
    if isinstance(item, ToolCallItem):
        if item.title:
            return f"Tool: {item.title}"
        if item.description:
            return f"Tool: {item.description}"
        ri = item.raw_item
        if isinstance(ri, dict):
            n = ri.get("name")
            if isinstance(n, str) and n:
                return f"Tool: {n}"
        n = getattr(ri, "name", None)
        if isinstance(n, str) and n:
            return f"Tool: {n}"
        return "Tool call"
    if isinstance(item, ToolCallOutputItem):
        return "Tool output"
    return event.name.replace("_", " ")


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
) -> None:
    """Consume ``stream.stream_events()``; with ``stream_text=True`` print ``ResponseTextDeltaEvent`` text."""
    stream_line_open = False

    def _end_stream_line_if_needed() -> None:
        nonlocal stream_line_open
        if stream_line_open:
            console.print()
            stream_line_open = False

    async for event in stream.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if stream_text and isinstance(event.data, ResponseTextDeltaEvent):
                console.print(event.data.delta, end="", highlight=False)
                stream_line_open = True
        elif isinstance(event, AgentUpdatedStreamEvent):
            agent = event.new_agent
            name = getattr(agent, "name", None) or type(agent).__name__
            _write_stream_status(f"Agent: {name}")
        elif isinstance(event, RunItemStreamEvent):
            _write_stream_status(_format_run_item_status(event))

    _finish_stream_status()
    _end_stream_line_if_needed()


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str | RunState[Any, Any],
    console: Console,
    *,
    stream_text: bool = False,
) -> RunResultStreaming:
    """Run one streamed turn. Gated tools pause inside tool invoke (Deepx HITL), not SDK interruptions."""
    stream = binding.run_streamed(inp)
    await drain_stream(stream, console, stream_text=stream_text)
    return stream


async def _stream_turn(
    binding: DeepRunBinding, user_input: str, console: Console
) -> None:
    await run_stream_until_settled(binding, user_input, console, stream_text=True)


def run_chat_stream(
    runner: DeepAgentRunner,
    *,
    session_id: str | None = None,
) -> None:
    """Interactive loop with token streaming. Uses ``--session`` from argv when ``session_id`` is omitted."""
    sid = parse_cli_session_arg() if session_id is None else session_id
    asyncio.run(
        run_interactive_repl(
            runner,
            session_id=sid,
            run_turn=_stream_turn,
        )
    )


__all__ = [
    "drain_stream",
    "run_chat_stream",
    "run_stream_until_settled",
]
