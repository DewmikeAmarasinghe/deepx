"""Terminal Human-in-the-Loop policy for :class:`deepx.middleware.hitl.Hitl`."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

from deepx.middleware.hitl import Hitl, HitlDecision, HitlRequest


def create_terminal_hitl(console: Console) -> Hitl:
    """Return a :class:`~deepx.middleware.hitl.Hitl` with an interactive terminal policy.

    The user is presented with three choices per tool invocation:

    1. **Reject** — block this call.
    2. **Allow once** — permit this single invocation.
    3. **Allow always** — skip approval for this tool name for the rest of the session.
    """

    async def policy(req: HitlRequest) -> HitlDecision:
        def sync_prompt() -> HitlDecision:
            raw = req.arguments_json or ""
            if raw.strip():
                try:
                    body = json.dumps(json.loads(raw), indent=3, ensure_ascii=False)
                except json.JSONDecodeError:
                    body = raw
            else:
                body = "(no arguments)"

            console.print(
                Panel(
                    body,
                    title=f"{req.agent_name} · approval · {req.tool_name}",
                    title_align="left",
                    border_style="yellow",
                    highlight=False,
                )
            )
            console.print("  [1] Reject")
            console.print("  [2] Allow once")
            console.print("  [3] Allow for rest of session (this tool)")

            while True:
                choice = input("Choice [1-3]: ").strip().lower()
                if not choice:
                    console.print("[dim](type 1, 2, or 3)[/dim]")
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