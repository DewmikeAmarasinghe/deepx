from __future__ import annotations

from typing import Any

from agents.result import RunResultStreaming
from agents.run_state import RunState
from rich.console import Console

from deepx.factory import DeepRunBinding

from deepx_cli.approvals import apply_choices_to_state


async def drain_stream(stream: RunResultStreaming, console: Console) -> None:
    try:
        from openai.types.responses import ResponseTextDeltaEvent
    except ImportError:
        ResponseTextDeltaEvent = None  # type: ignore[assignment,misc]

    async for event in stream.stream_events():
        if event.type == "raw_response_event" and ResponseTextDeltaEvent is not None:
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                console.print(data.delta, end="", highlight=False)
                continue
        if event.type == "run_item_stream_event":
            name = getattr(event.item, "type", None) or ""
            if name in ("tool_call_item", "tool_call_output_item", "message_output_item"):
                console.print(f"[dim]· {name}[/dim]")


async def run_stream_until_settled(
    binding: DeepRunBinding, inp: str | RunState[Any, Any], console: Console
) -> RunResultStreaming:
    stream = binding.run_streamed(inp)
    await drain_stream(stream, console)
    while stream.interruptions:
        console.print()
        state = stream.to_state()
        apply_choices_to_state(state, stream.interruptions, console)
        stream = binding.run_streamed(state)
        await drain_stream(stream, console)
    return stream


__all__ = ["drain_stream", "run_stream_until_settled"]
