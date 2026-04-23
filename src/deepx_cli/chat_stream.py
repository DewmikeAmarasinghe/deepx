from __future__ import annotations

from typing import Any

from agents.result import RunResultStreaming
from agents.run_state import RunState
from rich.console import Console

from deepx.factory import (
    DeepRunBinding,
    _cleanup_mcp_servers,
    _ensure_mcp_servers_connected,
)


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
) -> None:
    """Consume ``stream.stream_events()``; with ``stream_text=True`` print ``ResponseTextDeltaEvent`` text."""
    try:
        from openai.types.responses import ResponseTextDeltaEvent
    except ImportError:
        ResponseTextDeltaEvent = None  # type: ignore[assignment,misc]

    stream_line_open = False

    def _end_stream_line_if_needed() -> None:
        nonlocal stream_line_open
        if stream_line_open:
            console.print()
            stream_line_open = False

    async for event in stream.stream_events():
        if event.type != "raw_response_event":
            continue
        data = event.data
        if (
            stream_text
            and ResponseTextDeltaEvent is not None
            and isinstance(data, ResponseTextDeltaEvent)
        ):
            console.print(data.delta, end="", highlight=False)
            stream_line_open = True

    _end_stream_line_if_needed()


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str | RunState[Any, Any],
    console: Console,
    *,
    stream_text: bool = False,
) -> RunResultStreaming:
    """Run one streamed turn. Gated tools pause inside tool invoke (Deepx HITL), not SDK interruptions."""
    await _ensure_mcp_servers_connected(binding.agent)
    try:
        stream = binding.run_streamed(inp)
        await drain_stream(stream, console, stream_text=stream_text)
        return stream
    finally:
        await _cleanup_mcp_servers(binding.agent)


__all__ = [
    "drain_stream",
    "run_stream_until_settled",
]
