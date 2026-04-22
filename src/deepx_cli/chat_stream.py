from __future__ import annotations

from typing import Any

from agents.result import RunResultStreaming
from agents.run_state import RunState
from rich.console import Console

from deepx.factory import DeepRunBinding


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
    verbose_tools: bool = False,
) -> None:
    """Consume ``stream.stream_events()`` from the OpenAI Agents SDK.

    **Default** (``stream_text=True``, ``verbose_tools=False``): prints only assistant **text
    deltas** from ``raw_response_event`` payloads (e.g. ``ResponseTextDeltaEvent``). Tool calls
    and tool outputs are **not** shown.

    **With** ``verbose_tools=True``: also prints a dim line per ``run_item_stream_event`` for
    ``tool_call_item`` (name + arguments preview) and ``tool_call_output_item`` (truncated output).
    """
    try:
        from openai.types.responses import ResponseTextDeltaEvent
    except ImportError:
        ResponseTextDeltaEvent = None  # type: ignore[assignment,misc]

    async for event in stream.stream_events():
        if (
            event.type == "raw_response_event"
            and stream_text
            and ResponseTextDeltaEvent is not None
        ):
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                console.print(data.delta, end="", highlight=False)
                continue
        if event.type != "run_item_stream_event":
            continue
        item = event.item
        name = getattr(item, "type", None) or ""
        if verbose_tools and name == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            if isinstance(raw, dict):
                preview = f"{raw.get('name', '')} {raw.get('arguments', '')}"
            else:
                preview = f"{getattr(raw, 'name', '')} {getattr(raw, 'arguments', '')}"
            preview = (preview or "").replace("\n", " ")
            console.print(f"[dim]· tool_call[/dim] {preview}")
        elif verbose_tools and name == "tool_call_output_item":
            raw = getattr(item, "raw_item", None)
            if isinstance(raw, dict):
                out = raw.get("output", "")
            else:
                out = getattr(raw, "output", "")
            text = (str(out) if out is not None else "").replace("\n", " ")
            console.print(f"[dim]· tool_output[/dim] {text}")


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str | RunState[Any, Any],
    console: Console,
    *,
    stream_text: bool = False,
    verbose_tools: bool = False,
) -> RunResultStreaming:
    """Run one streamed turn. Gated tools pause inside tool invoke (Deepx HITL), not SDK interruptions."""
    stream = binding.run_streamed(inp)
    await drain_stream(
        stream, console, stream_text=stream_text, verbose_tools=verbose_tools
    )
    return stream


__all__ = [
    "drain_stream",
    "run_stream_until_settled",
]
