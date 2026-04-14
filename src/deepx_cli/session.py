from __future__ import annotations

import asyncio
import os
import uuid

from deepx.factory import DeepAgentRunner


def run_interactive(
    runner: DeepAgentRunner,
    *,
    session_id: str | None = None,
    user_name: str = "You",
    resume_hint: str | None = None,
) -> None:
    """Run an interactive multi-turn session in the terminal.

    Uses one asyncio event loop for the whole session so async tools (e.g. httpx) do not
    break across turns. Do not use run_sync() in a loop — it calls asyncio.run() per turn.

    Pass a session_id (or set SESSION_ID env var) to resume conversation history.

    Command:
        /bye — end the session (parsed locally; never sent to the agent).

    Args:
        runner: A configured DeepAgentRunner instance.
        session_id: Optional session ID. If omitted, uses SESSION_ID env var or generates one.
        user_name: Label shown before each user prompt. Defaults to "You".
        resume_hint: Exact shell command prefix to resume (e.g. ``python /path/to/script.py``);
            session id is appended in the printed line when helpful.
    """
    agent_label = runner._agent_name.replace("_", " ").title()
    sid = session_id or os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]

    if resume_hint:
        print(f"\n[Session: {sid}]  To resume later: `{resume_hint} {sid}`\n")
    else:
        print(f"\n[Session: {sid}]  Run `python <script> {sid}` to resume this session later.\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    turn = 0
    try:
        while True:
            try:
                user_input = input(f"{user_name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input == "/bye":
                print("Goodbye.")
                break

            coro = runner.run(user_input, session_id=sid, resume=(turn > 0))
            result = loop.run_until_complete(coro)
            print(f"\n{agent_label}: {result.output}\n")
            turn += 1
    finally:
        loop.close()
        asyncio.set_event_loop(None)
