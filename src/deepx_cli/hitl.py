"""Terminal policy for :class:`deepx.middleware.hitl.Hitl` (Rich + blocking input)."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

from deepx.middleware.hitl import Hitl, HitlDecision, HitlRequest


def create_terminal_hitl(console: Console) -> Hitl:
    """Reject / allow once / allow always (sticky per tool name for this :class:`Hitl` instance)."""

    async def policy(req: HitlRequest) -> HitlDecision:
        def sync_prompt() -> HitlDecision:
            console.print("\n")
            raw = req.arguments_json or ""
            if raw.strip():
                try:
                    parsed = json.loads(raw)
                    body = json.dumps(parsed, indent=3, ensure_ascii=False)
                except json.JSONDecodeError:
                    body = raw
            else:
                body = "(no arguments)"
            title = f"{req.agent_name} · approval · {req.tool_name}"
            console.print(
                Panel(
                    body,
                    title=title,
                    title_align="left",
                    border_style="yellow",
                    highlight=False,
                )
            )
            console.print("  [1] Reject")
            console.print("  [2] Allow once")
            console.print("  [3] Allow for rest of this session (this tool name)")
            while True:
                choice = input("Choice [1-3]: ").strip().lower()
                if not choice:
                    console.print("[dim](empty — type 1, 2, or 3)[/dim]")
                    continue
                if choice in ("1", "r", "reject", "n", "no"):
                    return HitlDecision.REJECT
                if choice in ("2", "o", "once", "y", "yes"):
                    return HitlDecision.ALLOW_ONCE
                if choice in ("3", "a", "always", "session"):
                    return HitlDecision.ALLOW_ALWAYS
                console.print("Enter 1, 2, or 3.")

        return await asyncio.to_thread(sync_prompt)

    return Hitl(policy)


__all__ = ["create_terminal_hitl"]
