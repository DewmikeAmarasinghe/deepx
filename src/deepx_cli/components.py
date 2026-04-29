from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


@dataclass(slots=True)
class ThinkingStream:
    content: str = ""

    def append(self, delta: str) -> None:
        self.content += delta

    def render(self) -> RenderableType:
        return Panel(
            Text(self.content or "…", style="dim"),
            title="thinking",
            border_style="magenta",
        )


@dataclass(slots=True)
class AssistantStream:
    agent_name: str
    content: str = ""

    def append(self, delta: str) -> None:
        self.content += delta

    def render(self) -> RenderableType:
        try:
            body: RenderableType = Markdown(self.content or " ")
        except Exception:
            body = Text(self.content or " ")
        return Panel(
            body,
            title=self.agent_name,
            border_style="green",
        )


@dataclass(slots=True)
class ToolInvocation:
    agent_name: str
    tool_name: str
    arguments: Any

    def render(self) -> RenderableType:
        try:
            body = json.dumps(self.arguments, indent=3, ensure_ascii=False)
        except TypeError:
            body = json.dumps(str(self.arguments), indent=3, ensure_ascii=False)

        return Panel(
            body,
            title=f"{self.agent_name} · {self.tool_name}",
            title_align="left",
            border_style="cyan",
            highlight=False,
        )


@dataclass(slots=True)
class HitlPanel:
    agent_name: str
    tool_name: str
    body: str

    def render(self) -> RenderableType:
        return Panel(
            self.body,
            title=f"{self.agent_name} · approval · {self.tool_name}",
            title_align="left",
            border_style="yellow",
            highlight=False,
        )


@dataclass(slots=True)
class ConversationStack:
    items: list[RenderableType] = field(default_factory=list)

    def push(self, item: RenderableType) -> None:
        self.items.append(item)

    def replace_last(self, item: RenderableType) -> None:
        if self.items:
            self.items[-1] = item
        else:
            self.items.append(item)

    def render(self) -> RenderableType:
        if not self.items:
            return Text("")
        return Group(*self.items)


__all__ = [
    "AssistantStream",
    "ConversationStack",
    "HitlPanel",
    "ThinkingStream",
    "ToolInvocation",
]
