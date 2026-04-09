from __future__ import annotations

import os
import uuid

from deepx.factory import DeepAgentRunner


def run_interactive(runner: DeepAgentRunner, *, session_id: str | None = None) -> None:
    """Run an interactive multi-turn session in the terminal.

    Maintains a single session_id across turns so the agent retains filesystem state.
    Set the SESSION_ID environment variable to resume a previous session.

    Args:
        runner: A configured DeepAgentRunner instance.
        session_id: Optional session ID. If omitted, uses SESSION_ID env var or generates one.
    """
    sid = session_id or os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    print(f"Session: {sid}  (set SESSION_ID={sid} to resume later)")
    print("Type 'exit' or press Ctrl-C to quit.\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        result = runner.run_sync(user_input, session_id=sid, resume=(turn > 0))
        print(f"\nAgent: {result.output}\n")
        turn += 1
