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

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _REPO_ROOT
SKILLS_DIR = DEMO_DIR / "skills"

_DEMO_BACKEND = LocalShellBackend(REPO_ROOT)
_AGENT_DBS = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_WEB_DB = str(_AGENT_DBS / "web_agent.db")

web_agent_runner = create_deep_agent(
    name="web_agent",
    description=(
        "Web research specialist: uses Tavily CLI (`tvly`) per skills under "
        "`test_demo/skills/tavily`; saves structured notes and markdown under the project tree."
    ),
    tools=None,
    skills=[
        str(SKILLS_DIR / "tavily"),
        str(SKILLS_DIR / "arxiv-search"),
    ],
    system_prompt=(
        "You are the **web_agent** internal service. **Assume the Tavily CLI (`tvly`) is installed** "
        "on the host; you have **no** Tavily HTTP API tools in-process.\n\n"
        "**Skills first:** use **`read_file`** on the relevant **tavily** skills and **arxiv-search** entries from "
        "your skills catalog (e.g. `tavily-cli/SKILL.md`, arXiv skill) before running commands.\n\n"
        "**Web research path:** use **`execute`** with a **`commands`** list (up to 5 parallel shell "
        "strings) to run **`tvly ...`** (add **`--json`** when you need structured data). Do **not** use "
        "`curl`, ad-hoc Python scraping, **BeautifulSoup**, or generic `requests` HTML parsing for "
        "open-web work — Tavily (+ arXiv skill where relevant) is the supported stack.\n\n"
        "Run **`tvly --status`** (or follow the skill) to confirm the CLI is logged in; if auth fails, "
        "say so briefly and stop.\n\n"
        "**Planning:** for any multi-step brief, **`write_todos` is mandatory** after skilling up; "
        "refresh the list as steps complete.\n\n"
        "Oversized **`execute`** output may be evicted to **`/_outputs/large_tool_results/`** — use "
        "**`read_file`** on the path from the tool message.\n\n"
        "For human-facing deliverables, use **`write_file`** under **`/_outputs/`**.\n\n"
        "Return final **artifact paths** plus a **tight summary** only — never paste large reports or raw CLI dumps."
    ),
    backend=_DEMO_BACKEND,
    checkpointer=_WEB_DB,
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)
