"""Optional Textual TUI shell (``uv sync --extra demo``).

This is a lightweight placeholder inspired by deepagents-cli: a clocked header/footer and quit
binding. Full agent chat, approvals, and tool panels remain in :mod:`deepx_cli.session` (Rich);
this module establishes an optional Textual entry point for future widgets.
"""

from __future__ import annotations


def main() -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer, Header, Static
    except ImportError as e:
        raise SystemExit(
            "Textual is not installed. Sync demo extras: uv sync --extra demo"
        ) from e

    class DeepxTui(App[None]):
        """Minimal Textual host for future HITL / tool panels."""

        BINDINGS = [("q", "quit", "Quit")]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(
                "Deepx Textual shell — use deepx_cli.session.run_chat for the full agent loop."
            )
            yield Footer()

        async def action_quit(self) -> None:
            self.exit()

    DeepxTui().run()


if __name__ == "__main__":
    main()
