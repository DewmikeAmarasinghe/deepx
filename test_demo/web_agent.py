"""Web research subagent: Tavily via host CLI + skills (demo)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.factory import create_deep_agent  # noqa: E402

load_dotenv()

REPO_ROOT = _REPO_ROOT

_DEMO_BACKEND = LocalShellBackend(REPO_ROOT)
_AGENT_DBS = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_WEB_DB = str(_AGENT_DBS / "web_agent.db")

web_agent_runner = create_deep_agent(
    name="web_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "Open-web research and reporting specialist: runs the **Tavily CLI (`tvly`)** under "
        "`test_demo/skills/tavily` and follows **`write-report`** for final written deliverables "
        "(structure, tone, sections). Use for live pages, news, docs, and long-form markdown. "
        "Requires a logged-in **`tvly`** (see `tavily-cli` skill)."
    ),
    tools=None,
    skills=[
        "./test_demo/skills/tavily",
        "./test_demo/skills/write-report",
    ],
    system_prompt=(
        "You are the **web_agent** internal service. **Assume the Tavily CLI (`tvly`) is installed** "
        "on the host; you have **no** Tavily HTTP API tools in-process.\n\n"
        "**Skills first:** use **`read_file`** on the relevant **tavily** skills (e.g. "
        "`tavily-cli/SKILL.md`) and on **`write-report/SKILL.md`** before planning multi-step work. "
        "Apply **write-report** standards to any report, memo, analysis, or stakeholder-facing "
        "markdown you produce.\n\n"
        "**Web research path:** use **`execute`** with a single **`command`** string to run **`tvly ...`** "
        "(add **`--json`** when you need structured data). Do **not** use "
        "`curl`, ad-hoc Python scraping, **BeautifulSoup**, or generic `requests` HTML parsing for "
        "open-web work — **Tavily is the supported stack** for fetching and researching the public web.\n\n"
        "Run **`tvly --status`** (or follow the skill) to confirm the CLI is logged in; if auth fails, "
        "say so briefly and stop.\n\n"
        "**Planning:** for any multi-step brief, **`write_todos` is mandatory** after skilling up; "
        "refresh the list as steps complete.\n\n"
        "Oversized **`execute`** output may be evicted to **`/_outputs/large_tool_results/`** — use "
        "**`read_file`** on the path from the tool message.\n\n"
        "For human-facing deliverables, use **`write_file`** under **`/_outputs/`**, following "
        "**write-report** structure (Executive Summary through Sources) unless the user specifies otherwise.\n\n"
        "Return final **artifact paths** plus a **tight summary** only — never paste large reports or raw CLI dumps."
    ),
    backend=_DEMO_BACKEND,
    checkpointer=_WEB_DB,
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)


def main() -> None:
    from deepx_cli.cli import run_interactive_cli

    run_interactive_cli(
        web_agent_runner,
        description="Deepx web / Tavily specialist demo (terminal).",
    )


if __name__ == "__main__":
    main()
