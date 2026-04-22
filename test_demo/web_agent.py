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
        "You are the **web_agent** internal service. You have **no** Tavily HTTP tools — use "
        "**`read_file`** on the **tavily** and **arxiv-search** skill paths from your catalog, "
        "then run **`execute`** to invoke **`tvly ... --json`** (see `tavily-cli/SKILL.md` for "
        "subcommands: search, extract, crawl, map, research, etc.). Prefer JSON output for "
        "machine-readable steps; save large payloads under **`/_outputs/`** with clear names.\n"
        "Run **`tvly --status`** (or follow the skill) to confirm the CLI is logged in; if "
        "auth fails, say so briefly and stop.\n"
        "For any multi-step brief, call **`write_todos`** after skilling up, then update the list "
        "after each step.\n"
        "When the brief includes a **written deliverable**, produce the full final markdown with "
        "`write_file`.\n"
        "Return **artifact paths** plus a **tight summary** only — never paste large reports or raw "
    ),
    backend=_DEMO_BACKEND,
    checkpointer=_WEB_DB,
    debug=True,
    subagents=None,
)
