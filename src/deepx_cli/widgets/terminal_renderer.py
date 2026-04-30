from __future__ import annotations

from rich.console import Console
from rich.live import Live

from deepx_cli.widgets.components import (
    AssistantStream,
    ConversationStack,
    ThinkingStream,
    ToolInvocation,
)


class TerminalRenderer:
    """Single Rich Live session: push thinking / assistant / tool blocks like a component tree."""

    def __init__(self, console: Console, *, refresh_per_second: int = 16) -> None:
        self.console = console
        self._stack = ConversationStack()
        self._thinking: ThinkingStream | None = None
        self._assistant: AssistantStream | None = None
        self._refresh_per_second = refresh_per_second
        self._live: Live | None = None

    def __enter__(self) -> TerminalRenderer:
        self._live = Live(
            self._stack.render(),
            console=self.console,
            refresh_per_second=self._refresh_per_second,
            auto_refresh=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None
        self._thinking = None
        self._assistant = None

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._stack.render(), refresh=True)

    def push_thinking_delta(self, delta: str) -> None:
        if self._assistant is not None:
            self._assistant = None
        if self._thinking is None:
            self._thinking = ThinkingStream()
            self._stack.push(self._thinking.render())
        self._thinking.append(delta)
        self._stack.replace_last(self._thinking.render())
        self._refresh()

    def push_assistant_delta(self, agent_name: str, delta: str) -> None:
        if self._thinking is not None:
            self._thinking = None
        if self._assistant is not None and self._assistant.agent_name != agent_name:
            self._assistant = None
        if self._assistant is None:
            self._assistant = AssistantStream(agent_name)
            self._stack.push(self._assistant.render())
        self._assistant.append(delta)
        self._stack.replace_last(self._assistant.render())
        self._refresh()

    def add_tool_call(self, agent_name: str, tool_name: str, arguments: object) -> None:
        self._thinking = None
        self._assistant = None
        self._stack.push(
            ToolInvocation(
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=arguments,
            ).render(),
        )
        self._refresh()

    def add_assistant_block(self, agent_name: str, text: str) -> None:
        """One-shot assistant panel (fallback final or non-streamed message)."""
        self._thinking = None
        self._assistant = None
        block = AssistantStream(agent_name, text)
        self._stack.push(block.render())
        self._refresh()


__all__ = ["TerminalRenderer"]
