from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agents import ItemHelpers
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.result import RunResultStreaming
from agents.run_state import RunState
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

from deepx.factory import (
    DeepAgentRunner,
    DeepRunBinding,
    interactive_stream_consumer,
)
from deepx_cli.session import parse_cli_session_arg, run_interactive_repl
from deepx_cli.widgets import (
    agent_label,
    print_tool_call_panel,
    tool_name_and_arguments,
)


def _raw_output_text_delta(data: Any) -> str | None:
    if isinstance(data, ResponseTextDeltaEvent):
        return data.delta or None
    if getattr(data, "type", None) == "response.output_text.delta":
        d = getattr(data, "delta", None)
        if isinstance(d, str):
            return d
    return None


def _raw_reasoning_delta(data: Any) -> str | None:
    """Visible reasoning summary / reasoning stream deltas (GPT-5 family, Responses API)."""
    if isinstance(data, ResponseReasoningSummaryTextDeltaEvent):
        return data.delta or None
    if isinstance(data, ResponseReasoningTextDeltaEvent):
        return data.delta or None
    t = getattr(data, "type", None)
    if t == "response.reasoning_summary_text.delta":
        d = getattr(data, "delta", None)
        if isinstance(d, str):
            return d
    if t == "response.reasoning_text.delta":
        d = getattr(data, "delta", None)
        if isinstance(d, str):
            return d
    if t == "response.reasoning_summary.delta":
        d = getattr(data, "delta", None)
        if isinstance(d, str):
            return d
    return None


@dataclass
class _StreamUIState:
    assistant_line_open: bool = False
    streamed_assistant_text: bool = False
    message_emitted: bool = False
    current_agent: str = ""
    last_prefixed_prose_agent: str | None = None
    reasoning_active: bool = False


class _StreamEventPrinter:
    """Rich UI: bordered tool panels, lowercase agent labels before prose segments."""

    def __init__(self, console: Console, *, root_agent_name: str) -> None:
        self._console = console
        self._state = _StreamUIState(current_agent=root_agent_name.strip() or "agent")

    def _end_assistant_line_if_needed(self) -> None:
        if self._state.assistant_line_open:
            self._console.print()
            self._state.assistant_line_open = False

    def _finish_reasoning_block(self) -> None:
        if self._state.reasoning_active:
            self._console.print()
            self._state.reasoning_active = False

    def _ensure_prose_leader(self) -> None:
        """Print ``agent:`` once before the next assistant prose for this agent."""
        self._finish_reasoning_block()
        ca = self._state.current_agent
        if self._state.last_prefixed_prose_agent == ca:
            return
        self._end_assistant_line_if_needed()
        self._console.print(f"{ca}:")
        self._state.last_prefixed_prose_agent = ca

    def _emit_raw(self, data: Any) -> None:
        delta = _raw_output_text_delta(data)
        if not delta:
            return
        self._finish_reasoning_block()
        self._ensure_prose_leader()
        self._console.print(delta, end="", highlight=False)
        self._state.assistant_line_open = True
        self._state.streamed_assistant_text = True

    def _emit_reasoning_from_raw(self, data: Any) -> None:
        delta = _raw_reasoning_delta(data)
        if not delta:
            return
        self._end_assistant_line_if_needed()
        if not self._state.reasoning_active:
            self._console.print("[dim]thinking[/dim]")
            self._state.reasoning_active = True
        self._console.print(delta, end="", style="dim")

    def _emit_raw_payload(self, data: Any) -> None:
        """Single Responses ``data`` object: reasoning deltas take precedence, else output text."""
        if _raw_reasoning_delta(data) is not None:
            self._emit_reasoning_from_raw(data)
            return
        self._emit_raw(data)

    def _emit_tool_call(self, item: ToolCallItem) -> None:
        aname = agent_label(item.agent)
        tname, args = tool_name_and_arguments(item)
        self._finish_reasoning_block()
        self._end_assistant_line_if_needed()
        print_tool_call_panel(self._console, aname, tname, args, border_style="cyan")

    def handle(self, event: Any) -> None:
        if isinstance(event, RawResponsesStreamEvent):
            self._emit_raw_payload(event.data)
        elif isinstance(event, AgentUpdatedStreamEvent):
            name = agent_label(event.new_agent)
            if name != self._state.current_agent:
                self._state.current_agent = name
                self._state.last_prefixed_prose_agent = None
        elif isinstance(event, RunItemStreamEvent):
            if event.name == "tool_output":
                return
            if isinstance(event.item, ToolCallOutputItem):
                return
            if event.name == "message_output_created" and isinstance(
                event.item, MessageOutputItem
            ):
                if not self._state.streamed_assistant_text:
                    text = ItemHelpers.text_message_output(event.item) or ""
                    if text:
                        self._finish_reasoning_block()
                        self._ensure_prose_leader()
                        self._console.print(text, end="", highlight=False)
                        self._state.assistant_line_open = True
                        self._state.message_emitted = True
                return
            if isinstance(event.item, ToolCallItem):
                self._emit_tool_call(event.item)

    def emit_fallback_final(self, text: str) -> None:
        if not text.strip():
            return
        self._finish_reasoning_block()
        self._ensure_prose_leader()
        self._console.print(text)

    def finalize(self) -> None:
        self._finish_reasoning_block()
        self._end_assistant_line_if_needed()

    def assistant_body_emitted(self) -> bool:
        return self._state.streamed_assistant_text or self._state.message_emitted


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
    root_agent_name: str = "agent",
) -> None:
    """Consume ``stream.stream_events()``; with ``stream_text=True`` print assistant + tool progress."""
    printer = _StreamEventPrinter(console, root_agent_name=root_agent_name)

    async for event in stream.stream_events():
        if not stream_text and isinstance(event, RawResponsesStreamEvent):
            continue
        printer.handle(event)

    printer.finalize()


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str | RunState[Any, Any],
    console: Console,
    *,
    stream_text: bool = False,
) -> RunResultStreaming:
    """Run one streamed turn. Nested subagent ``run_streamed`` calls share ``interactive_stream_consumer``."""
    root = binding._runner._agent_name
    printer = _StreamEventPrinter(console, root_agent_name=root)

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
        if fo is not None and str(fo).strip() and not printer.assistant_body_emitted():
            printer.emit_fallback_final(str(fo))
        return stream
    finally:
        interactive_stream_consumer.reset(token)


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
