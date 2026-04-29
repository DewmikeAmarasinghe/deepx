from __future__ import annotations

import asyncio
import types
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
from deepx_cli.utils import agent_label, tool_name_and_arguments
from deepx_cli.components import AssistantStream, ToolInvocation
from deepx_cli.terminal_renderer import TerminalRenderer


def _raw_output_text_delta(data: Any) -> str | None:
    if isinstance(data, ResponseTextDeltaEvent):
        return data.delta or None
    if getattr(data, "type", None) == "response.output_text.delta":
        d = getattr(data, "delta", None)
        if isinstance(d, str):
            return d
    return None


def _raw_reasoning_delta(data: Any) -> str | None:
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
class _StreamState:
    streamed_assistant_text: bool = False
    message_emitted: bool = False
    current_agent: str = ""
    reasoning_active: bool = False


class _StreamLayout:
    """One assistant turn: Live stack when streaming, plain Rich renderables otherwise."""

    def __init__(
        self,
        console: Console,
        *,
        root_agent_name: str,
        use_live: bool,
    ) -> None:
        self._console = console
        self._use_live = use_live
        self._live: TerminalRenderer | None = (
            TerminalRenderer(console) if use_live else None
        )
        self._state = _StreamState(current_agent=root_agent_name.strip() or "agent")

    def __enter__(self) -> _StreamLayout:
        if self._live is not None:
            self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def _end_reasoning_block(self) -> None:
        if self._state.reasoning_active and self._live is None:
            self._console.print()
        self._state.reasoning_active = False

    def _emit_reasoning_from_raw(self, data: Any) -> None:
        delta = _raw_reasoning_delta(data)
        if not delta:
            return
        if not self._state.reasoning_active:
            self._state.reasoning_active = True
        if self._live is not None:
            self._live.push_thinking_delta(delta)
        else:
            self._console.print(delta, end="", style="dim")

    def _emit_raw(self, data: Any) -> None:
        delta = _raw_output_text_delta(data)
        if not delta:
            return
        self._end_reasoning_block()
        agent = self._state.current_agent
        if self._live is not None:
            self._live.push_assistant_delta(agent, delta)
        else:
            self._console.print(delta, end="", highlight=False)
        self._state.streamed_assistant_text = True

    def _emit_raw_payload(self, data: Any) -> None:
        if _raw_reasoning_delta(data) is not None:
            self._emit_reasoning_from_raw(data)
            return
        self._emit_raw(data)

    def _emit_tool_call(self, item: ToolCallItem) -> None:
        self._end_reasoning_block()
        aname = agent_label(item.agent)
        tname, args = tool_name_and_arguments(item)
        if self._live is not None:
            self._live.add_tool_call(aname, tname, args)
        else:
            self._console.print()
            self._console.print(ToolInvocation(aname, tname, args).render())

    def handle(self, event: Any) -> None:
        if isinstance(event, RawResponsesStreamEvent):
            self._emit_raw_payload(event.data)
        elif isinstance(event, AgentUpdatedStreamEvent):
            name = agent_label(event.new_agent)
            if name != self._state.current_agent:
                self._state.current_agent = name
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
                        self._end_reasoning_block()
                        agent = self._state.current_agent
                        if self._live is not None:
                            self._live.add_assistant_block(agent, text)
                        else:
                            self._console.print()
                            self._console.print(AssistantStream(agent, text).render())
                        self._state.message_emitted = True
                return
            if isinstance(event.item, ToolCallItem):
                self._emit_tool_call(event.item)

    def emit_fallback_final(self, text: str) -> None:
        if not text.strip():
            return
        self._end_reasoning_block()
        agent = self._state.current_agent
        if self._live is not None:
            self._live.add_assistant_block(agent, text)
        else:
            self._console.print()
            self._console.print(AssistantStream(agent, text).render())

    def assistant_body_emitted(self) -> bool:
        return self._state.streamed_assistant_text or self._state.message_emitted


async def drain_stream(
    stream: RunResultStreaming,
    console: Console,
    *,
    stream_text: bool = False,
    root_agent_name: str = "agent",
) -> None:
    """Consume ``stream.stream_events()``; with ``stream_text=True`` render assistant + tool progress."""
    layout = _StreamLayout(
        console, root_agent_name=root_agent_name, use_live=stream_text
    )
    with layout:
        async for event in stream.stream_events():
            if not stream_text and isinstance(event, RawResponsesStreamEvent):
                continue
            layout.handle(event)


async def run_stream_until_settled(
    binding: DeepRunBinding,
    inp: str | RunState[Any, Any],
    console: Console,
    *,
    stream_text: bool = False,
) -> RunResultStreaming:
    """Run one streamed turn. Nested subagent ``run_streamed`` calls share ``interactive_stream_consumer``."""
    root = binding._runner._agent_name
    layout = _StreamLayout(console, root_agent_name=root, use_live=stream_text)

    def forward(ev: Any) -> None:
        if not stream_text and isinstance(ev, RawResponsesStreamEvent):
            return
        layout.handle(ev)

    with layout:
        token = interactive_stream_consumer.set(forward)
        try:
            stream = binding.run_streamed(inp)
            async for event in stream.stream_events():
                forward(event)
            fo = stream.final_output
            if (
                fo is not None
                and str(fo).strip()
                and not layout.assistant_body_emitted()
            ):
                layout.emit_fallback_final(str(fo))
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
