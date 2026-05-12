"""Web research subagent: Tavily via host CLI + skills.

Run standalone::

    python test_demo/web_agent.py --chat
    python test_demo/web_agent.py --chat_sync
    python test_demo/web_agent.py --chat --session <id>
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.factory import create_deep_agent  # noqa: E402

web_sys = """\
You are the web research specialist. The Tavily CLI (`tvly`) is your only supported tool
for fetching the open web — do not use curl, requests, or BeautifulSoup.

## Before starting any task

Read the relevant skill files under `/test_demo/skills/` (tavily for web related tasks and write-report when drafting long reports).

Run `tvly --status` first to confirm authentication. If it fails, report the error and stop.

If the result is very large it will be written to `/_outputs/large_tool_results/` — read it from there
with `read_file`.

## Report structure

For every written deliverable, apply the write-report skill structure:
  Executive Summary → Background → Findings → Analysis → Recommendations → Sources

Omit sections only if the caller explicitly says so. Keep sections concise.

## Output

Choose the output path as follows:
- If the orchestrator specified a path in its brief, write there.
- Otherwise write to `/_outputs/<descriptive_slug>.md`.

Return only: the output file path(s) and a two-to-three sentence summary of the findings.
Do not paste the report body in the reply.
"""

web_agent_runner = create_deep_agent(
    name="web_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "Open-web research and reporting specialist. Runs the Tavily CLI (`tvly`) for live "
        "search, news, and documentation. Produces structured markdown reports under `/_outputs/` "
        "following the write-report skill. Requires a logged-in `tvly` on the host."
    ),
    tools=None,
    skills=[
        "./test_demo/skills/tavily",
        "./test_demo/skills/write-report",
    ],
    system_prompt=web_sys,
    backend=LocalShellBackend(_REPO_ROOT),
    checkpointer=str(_REPO_ROOT / "test_demo" / "dbs" / "agent_dbs" / "web_agent.db"),
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)


def main() -> None:
    from deepx_cli import run_interactive_cli
    run_interactive_cli(web_agent_runner, description="web_agent — Tavily research REPL.")


if __name__ == "__main__":
    (_REPO_ROOT / "test_demo" / "dbs" / "agent_dbs").mkdir(parents=True, exist_ok=True)
    main()