"""deepx_cli — terminal REPL and streaming UI for the deepx framework."""

from deepx_cli.cli import run_interactive_cli
from deepx_cli.run import run_chat_stream, run_chat_sync

__all__ = [
    "run_interactive_cli",
    "run_chat_stream",
    "run_chat_sync",
]