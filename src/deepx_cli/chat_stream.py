from __future__ import annotations

import json
from typing import Any

from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.result import RunResultStreaming
from agents.run_state import RunState
from rich.console import Console

from deepx.factory import DeepRunBinding


def _summarize_raw_event(data: Any, max_len: int = 140) -> str:
    ev_type = getattr(data, "type", None) or type(data).__name__
    try:
        body = repr(data)
    except Exception:
        body = f"<{type(data).__name__}>"
    body = body.replace("\n", " ")
    if len(body) > max_len:
        body = body[: max_len - 3] + "..."
    return f"{ev_type} {body}"


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
    verbose_tools: bool = False,
    verbose: bool = False,
) -> None:
    """Consume ``stream.stream_events()`` from the OpenAI Agents SDK.

    **Default** (``stream_text=True``): prints assistant **text deltas** from
    ``ResponseTextDeltaEvent`` payloads only.

    **``verbose_tools=True``** (or ``verbose=True``): prints dim lines for tool calls and outputs.

    **``verbose=True``**: also logs **agent switches** (``agent_updated_stream_event``), **run
    items** (tools, message completion), and **raw** stream events (but not per-character
    ``ResponseFunctionCallArgumentsDeltaEvent`` spam — args appear via tool/run items). Assistant prose uses
    **one** channel: when ``stream_text=True``, live deltas are the prose channel and completed
    ``message_output_item`` events are only summarized (char count) so text is not printed twice.
    ``ResponseTextDoneEvent`` raw events are skipped in that mode for the same reason. Other
    ``raw_response_event`` types emit a compact ``type`` + truncated ``repr`` line.
    """
    show_tools = verbose_tools or verbose

    try:
        from openai.types.responses import (
            ResponseFunctionCallArgumentsDeltaEvent,
            ResponseTextDeltaEvent,
            ResponseTextDoneEvent,
        )
    except ImportError:
        ResponseFunctionCallArgumentsDeltaEvent = None  # type: ignore[assignment,misc]
        ResponseTextDeltaEvent = None  # type: ignore[assignment,misc]
        ResponseTextDoneEvent = None  # type: ignore[assignment,misc]

    stream_line_open = False

    def _end_stream_line_if_needed() -> None:
        nonlocal stream_line_open
        if stream_line_open:
            console.print()
            stream_line_open = False

    async for event in stream.stream_events():
        et = event.type

        if et == "agent_updated_stream_event" and verbose:
            _end_stream_line_if_needed()
            na = getattr(event, "new_agent", None)
            nm = getattr(na, "name", None) if na is not None else None
            label = str(nm).strip() if nm else "agent"
            console.print(f"[dim]→ agent[/dim] [bold]{label}[/bold]")
            continue

        if et == "raw_response_event":
            data = event.data
            if stream_text and ResponseTextDeltaEvent is not None and isinstance(
                data, ResponseTextDeltaEvent
            ):
                console.print(data.delta, end="", highlight=False)
                stream_line_open = True
                continue
            if verbose:
                if (
                    stream_text
                    and ResponseTextDoneEvent is not None
                    and isinstance(data, ResponseTextDoneEvent)
                ):
                    continue
                if ResponseTextDeltaEvent is not None and isinstance(
                    data, ResponseTextDeltaEvent
                ):
                    continue
                if (
                    ResponseFunctionCallArgumentsDeltaEvent is not None
                    and isinstance(data, ResponseFunctionCallArgumentsDeltaEvent)
                ):
                    continue
                _end_stream_line_if_needed()
                console.print(f"[dim]· raw[/dim] {_summarize_raw_event(data)}")
            continue

        if et != "run_item_stream_event":
            continue

        item = event.item
        iname = getattr(item, "type", "") or ""
        ev_name = getattr(event, "name", "") or ""

        if verbose and iname == "message_output_item" and isinstance(
            item, MessageOutputItem
        ):
            body = ItemHelpers.text_message_output(item)
            if stream_text:
                _end_stream_line_if_needed()
                console.print(f"[dim]· message_output[/dim] ({len(body)} chars)")
            else:
                console.print(body, end="", highlight=False)
                console.print()
            continue

        if show_tools and iname == "tool_call_item" and isinstance(item, ToolCallItem):
            raw = item.raw_item
            if isinstance(raw, dict):
                preview = f"{raw.get('name', '')} {raw.get('arguments', '')}"
            else:
                preview = f"{getattr(raw, 'name', '')} {getattr(raw, 'arguments', '')}"
            preview = (preview or "").replace("\n", " ")
            tag = f"[dim]· {ev_name}[/dim]" if verbose else "[dim]· tool_call[/dim]"
            _end_stream_line_if_needed()
            console.print(f"{tag} {preview}")
            continue

        if show_tools and iname == "tool_call_output_item" and isinstance(
            item, ToolCallOutputItem
        ):
            out = item.output
            text = (str(out) if out is not None else "").replace("\n", " ")
            tag = f"[dim]· {ev_name}[/dim]" if verbose else "[dim]· tool_output[/dim]"
            _end_stream_line_if_needed()
            console.print(f"{tag} {text}")
            continue

        if verbose:
            _end_stream_line_if_needed()
            if isinstance(item, HandoffCallItem):
                raw = item.raw_item
                tn = getattr(raw, "name", "") if raw is not None else ""
                console.print(f"[dim]· handoff_call[/dim] {tn}")
            elif isinstance(item, HandoffOutputItem):
                console.print("[dim]· handoff_output[/dim]")
            else:
                extra = ""
                raw = getattr(item, "raw_item", None)
                if raw is not None:
                    try:
                        extra = json.dumps(raw, default=str)[:120]
                    except TypeError:
                        extra = str(raw)[:120]
                    if len(extra) > 120:
                        extra = extra[:117] + "..."
                console.print(f"[dim]· run_item[/dim] {iname} {ev_name} {extra}".rstrip())

    _end_stream_line_if_needed()


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str | RunState[Any, Any],
    console: Console,
    *,
    stream_text: bool = False,
    verbose_tools: bool = False,
    verbose: bool = False,
) -> RunResultStreaming:
    """Run one streamed turn. Gated tools pause inside tool invoke (Deepx HITL), not SDK interruptions."""
    stream = binding.run_streamed(inp)
    await drain_stream(
        stream,
        console,
        stream_text=stream_text,
        verbose_tools=verbose_tools,
        verbose=verbose,
    )
    return stream


__all__ = [
    "drain_stream",
    "run_stream_until_settled",
]
