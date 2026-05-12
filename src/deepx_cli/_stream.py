"""Internal streaming helpers for deepx_cli.

Not part of the public API — import from :mod:`deepx_cli.run` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents import ItemHelpers
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.result import RunResultStreaming
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_reasoning_text_delta_event import (
    ResponseReasoningTextDeltaEvent,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from deepx.factory import DeepRunBinding, interactive_stream_consumer


# ---------------------------------------------------------------------------
# Delta extractors
# ---------------------------------------------------------------------------


def _output_text_delta(data: Any) -> str | None:
    if isinstance(data, ResponseTextDeltaEvent):
        return data.delta or None
    if getattr(data, "type", None) == "response.output_text.delta":
        d = getattr(data, "delta", None)
        if isinstance(d, str):
            return d
    return None


def _reasoning_delta(data: Any) -> str | None:
    if isinstance(data, (ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningTextDeltaEvent)):
        return data.delta or None
    t = getattr(data, "type", None)
    for suffix in (
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary.delta",
    ):
        if t == suffix:
            d = getattr(data, "delta", None)
            if isinstance(d, str):
                return d
    return None


def _is_reasoning_section_done(data: Any) -> bool:
    return getattr(data, "type", None) in (
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done",
    )


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------


def _terminal_width(console: Console) -> int:
    return console.size.width or 120


def _agent_label(agent: Any) -> str:
    if agent is None:
        return "agent"
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return type(agent).__name__


def _print_tool_panel(
    console: Console,
    agent_name: str,
    tool_name: str,
    arguments: Any,
    *,
    border_style: str = "cyan",
) -> None:
    try:
        body = json.dumps(arguments, indent=3, ensure_ascii=False)
    except TypeError:
        body = json.dumps(str(arguments), indent=3, ensure_ascii=False)
    console.print(
        Panel(
            body,
            title=f"{agent_name} · {tool_name}",
            title_align="left",
            border_style=border_style,
            highlight=False,
            width=_terminal_width(console),
        )
    )


def _parse_tool_call(item: ToolCallItem) -> tuple[str, Any]:
    ri = item.raw_item
    if isinstance(ri, dict):
        tname = str(ri.get("name") or ri.get("call_id") or "?")
        raw_args: Any = ri.get("arguments", "{}")
    elif hasattr(ri, "model_dump"):
        d = ri.model_dump(mode="python", exclude_unset=True)
        tname = str(d.get("name", getattr(ri, "name", "?")))
        raw_args = d.get("arguments", getattr(ri, "arguments", "{}"))
    else:
        tname = str(getattr(ri, "name", None) or "?")
        raw_args = getattr(ri, "arguments", "{}")

    if isinstance(raw_args, str):
        if not raw_args.strip():
            return tname, {}
        try:
            return tname, json.loads(raw_args)
        except json.JSONDecodeError:
            return tname, {"_raw_arguments": raw_args}
    if raw_args is None:
        return tname, {}
    if isinstance(raw_args, dict):
        return tname, raw_args
    return tname, {"_value": raw_args}


# ---------------------------------------------------------------------------
# Event printer
# ---------------------------------------------------------------------------


@dataclass
class _UIState:
    assistant_line_open: bool = False
    streamed_assistant_text: bool = False
    message_emitted: bool = False
    current_agent: str = ""
    last_prefixed_prose_agent: str | None = None
    reasoning_active: bool = False


class _StreamEventPrinter:
    """Renders Rich UI: full-width tool panels; Live panels for token stream when enabled."""

    def __init__(
        self,
        console: Console,
        *,
        root_agent_name: str,
        rich_stream: bool = False,
    ) -> None:
        self._console = console
        self._rich_stream = rich_stream
        self._state = _UIState(current_agent=root_agent_name.strip() or "agent")
        self._reasoning_live: Live | None = None
        self._reasoning_buf = ""
        self._response_live: Live | None = None
        self._response_buf = ""

    def _stop_reasoning_live(self) -> None:
        if self._reasoning_live is not None:
            self._reasoning_live.__exit__(None, None, None)
            self._reasoning_live = None
        self._reasoning_buf = ""

    def _stop_response_live(self) -> None:
        if self._response_live is not None:
            self._response_live.__exit__(None, None, None)
            self._response_live = None
        self._response_buf = ""

    def _thinking_panel(self) -> Panel:
        raw = self._reasoning_buf.strip() or " "
        return Panel(
            Text(raw),
            title=f"{self._state.current_agent} · thinking",
            title_align="left",
            border_style="grey37",
            width=_terminal_width(self._console),
        )

    def _response_panel(self) -> Panel:
        raw = self._response_buf or " "
        return Panel(
            Text(raw),
            title=f"{self._state.current_agent} · response",
            title_align="left",
            border_style="green",
            width=_terminal_width(self._console),
        )

    def _response_panel_from_text(self, text: str) -> Panel:
        return Panel(
            Text(text.strip()),
            title=f"{self._state.current_agent} · response",
            title_align="left",
            border_style="green",
            width=_terminal_width(self._console),
        )

    def _close_assistant_line(self) -> None:
        if self._state.assistant_line_open:
            self._console.print()
            self._state.assistant_line_open = False

    def _close_reasoning_block(self) -> None:
        self._stop_reasoning_live()
        if self._state.reasoning_active:
            self._console.print()
            self._state.reasoning_active = False

    def _ensure_prose_leader(self) -> None:
        self._close_reasoning_block()
        ca = self._state.current_agent
        if self._state.last_prefixed_prose_agent == ca:
            return
        self._close_assistant_line()
        self._console.print(f"{ca}:")
        self._state.last_prefixed_prose_agent = ca

    def _emit_text_delta(self, data: Any) -> None:
        delta = _output_text_delta(data)
        if not delta:
            return
        self._close_reasoning_block()
        if self._rich_stream:
            if self._response_live is None:
                self._response_buf = ""
                live = Live(
                    self._response_panel(),
                    console=self._console,
                    auto_refresh=False,
                    vertical_overflow="visible",
                    transient=False,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                live.__enter__()
                self._response_live = live
            self._response_buf += delta
            self._response_live.update(self._response_panel(), refresh=True)
            self._state.streamed_assistant_text = True
        else:
            self._ensure_prose_leader()
            self._console.print(delta, end="", highlight=False)
            self._state.assistant_line_open = True
            self._state.streamed_assistant_text = True

    def _emit_reasoning_delta(self, data: Any) -> None:
        delta = _reasoning_delta(data)
        if not delta:
            return
        self._stop_response_live()
        if not self._rich_stream:
            self._close_assistant_line()
        if self._rich_stream:
            if self._reasoning_live is None:
                self._reasoning_buf = ""
                live = Live(
                    self._thinking_panel(),
                    console=self._console,
                    auto_refresh=False,
                    vertical_overflow="visible",
                    transient=False,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                live.__enter__()
                self._reasoning_live = live
            self._reasoning_buf += delta
            self._reasoning_live.update(self._thinking_panel(), refresh=True)
        else:
            if not self._state.reasoning_active:
                self._console.print(f"[dim]{self._state.current_agent} thinking[/dim]")
                self._state.reasoning_active = True
            self._console.print(delta, end="", style="dim")

    def _on_reasoning_section_done(self) -> None:
        if self._rich_stream:
            self._stop_reasoning_live()
        elif self._state.reasoning_active:
            self._console.print()
            self._state.reasoning_active = False

    def _emit_tool_call(self, item: ToolCallItem) -> None:
        tname, args = _parse_tool_call(item)
        self._close_reasoning_block()
        self._stop_response_live()
        if not self._rich_stream:
            self._close_assistant_line()
        _print_tool_panel(self._console, _agent_label(item.agent), tname, args)

    def handle(self, event: Any) -> None:
        if isinstance(event, RawResponsesStreamEvent):
            data = event.data
            if _is_reasoning_section_done(data):
                self._on_reasoning_section_done()
            elif _reasoning_delta(data) is not None:
                self._emit_reasoning_delta(data)
            else:
                self._emit_text_delta(data)

        elif isinstance(event, AgentUpdatedStreamEvent):
            name = _agent_label(event.new_agent)
            if name != self._state.current_agent:
                self._close_reasoning_block()
                self._stop_response_live()
                if not self._rich_stream:
                    self._close_assistant_line()
                self._state.current_agent = name
                self._state.last_prefixed_prose_agent = None

        elif isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallOutputItem):
                return
            if event.name == "message_output_created" and isinstance(
                event.item, MessageOutputItem
            ):
                if not self._state.streamed_assistant_text:
                    text = ItemHelpers.text_message_output(event.item) or ""
                    if text:
                        self._close_reasoning_block()
                        if self._rich_stream:
                            self._stop_response_live()
                            self._console.print(self._response_panel_from_text(text))
                            self._state.message_emitted = True
                        else:
                            self._ensure_prose_leader()
                            self._console.print(text, end="", highlight=False)
                            self._state.assistant_line_open = True
                            self._state.message_emitted = True
            elif isinstance(event.item, ToolCallItem):
                self._emit_tool_call(event.item)

    def emit_fallback_final(self, text: str) -> None:
        if not text.strip():
            return
        self._close_reasoning_block()
        if self._rich_stream:
            self._stop_response_live()
            self._console.print(self._response_panel_from_text(text))
        else:
            self._ensure_prose_leader()
            self._console.print(text)
        self._state.message_emitted = True

    def finalize(self) -> None:
        self._close_reasoning_block()
        self._stop_response_live()
        if not self._rich_stream:
            self._close_assistant_line()

    def body_emitted(self) -> bool:
        return self._state.streamed_assistant_text or self._state.message_emitted


# ---------------------------------------------------------------------------
# Drain / run helpers
# ---------------------------------------------------------------------------


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
    root_agent_name: str = "agent",
) -> None:
    printer = _StreamEventPrinter(
        console, root_agent_name=root_agent_name, rich_stream=stream_text
    )
    async for event in stream.stream_events():
        if not stream_text and isinstance(event, RawResponsesStreamEvent):
            continue
        printer.handle(event)
    printer.finalize()


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str,
    console: Console,
    *,
    stream_text: bool = False,
) -> RunResultStreaming:
    """Run one turn via deepx's streaming pipeline.

    Args:
        binding:     Bound runner for this session turn.
        inp:         User input string.
        console:     Rich console for output.
        stream_text: ``True`` → token stream inside bordered **Live** panels titled
                     ``{agent} · thinking`` / ``{agent} · response``, plus tool panels.
                     ``False`` → tool panels only (no token stream).
    """
    root = binding._runner._agent_name
    printer = _StreamEventPrinter(console, root_agent_name=root, rich_stream=stream_text)

    def forward(ev: Any) -> None:
        if not stream_text and isinstance(ev, RawResponsesStreamEvent):
            return
        printer.handle(ev)

    token = interactive_stream_consumer.set(forward)
    try:
        stream = binding.run_streamed(inp)
        async for event in stream.stream_events():
            forward(event)
        printer.finalize()
        fo = stream.final_output
        if fo is not None and str(fo).strip() and not printer.body_emitted():
            printer.emit_fallback_final(str(fo))
        return stream
    finally:
        interactive_stream_consumer.reset(token)