"""Multi-agent orchestrator demo.

Run from the repository root (the ``test_demo`` tree is not shipped in the wheel)::

    uv sync --extra demo
    python -m test_demo.orchestrator --chat
    python -m test_demo.orchestrator --chat --verbose
    python -m test_demo.orchestrator --chat_sync

``--chat`` (default) streams assistant tokens; ``--verbose`` adds SDK stream events (no effect on
``--chat_sync``, which has no token stream).

Installing the ``deepx`` distribution places ``deepx`` and ``deepx_cli`` on the path; this
orchestrator module is for local development and demos only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents import RunContextWrapper, function_tool  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from deepx import create_deep_agent  # noqa: E402
from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx.context import AgentContext  # noqa: E402
from test_demo.hf_agent import hf_agent_runner  # noqa: E402
from test_demo.pdf_agent import pdf_agent_runner  # noqa: E402
from test_demo.sql_agent import sql_agent_runner  # noqa: E402
from test_demo.web_agent import web_agent_runner  # noqa: E402

_RENDER_FILES_LINE_LIMIT = 10_000_000
_render_console = Console(highlight=False)


@function_tool
async def render_files(
    ctx: RunContextWrapper[AgentContext],
    paths: list[str],
) -> str:
    """Show finished artifacts to the human in the terminal.

    Call **once** when work is **complete** (or when the user should see final outputs), with
    **every** user-relevant path — e.g. final reports, key tables, or merged PDFs. Do **not** use
    this for exploratory reading while delegating; specialists return paths, and you consolidate
    before rendering.
    """
    sid = ctx.context.session_id
    parts: list[str] = []
    w = _render_console.size.width or 120
    for path in paths:
        p = (path or "").strip()
        if not p:
            continue
        rr = ctx.context.backend.read(sid, p, 0, _RENDER_FILES_LINE_LIMIT)
        if rr.error:
            _render_console.print(
                Panel(
                    rr.error,
                    title=f"render_files · {p}",
                    border_style="yellow",
                    expand=True,
                    width=w,
                )
            )
            parts.append(f"{p}: error")
            continue
        text = rr.content or ""
        _render_console.print(
            Panel(
                text,
                title=f"render_files · {p}",
                border_style="yellow",
                expand=True,
                width=w,
            )
        )
        parts.append(f"{p}: {len(text)} chars")
    return "Rendered: " + "; ".join(parts) if parts else "No paths provided."


DBS_DIR = REPO_ROOT / "test_demo" / "dbs"
TEST_DBS = DBS_DIR / "test_dbs"
AGENT_DBS = DBS_DIR / "agent_dbs"
for d in (TEST_DBS, AGENT_DBS):
    d.mkdir(parents=True, exist_ok=True)

DEMO_BACKEND = FilesystemBackend(REPO_ROOT)

ORCH_DB = str(AGENT_DBS / "orchestrator.db")

ORCH_SKILLS_DIR = REPO_ROOT / "test_demo" / "skills" / "orchestrate"
orch_tools = [render_files]

DEMO_ORCHESTRATOR_SYS = """\
## Role

You **only coordinate** specialists. You do **not** run web/Tavily, SQL, PDF, or HF work yourself.

**After a specialist tool returns:** trust its **short summary** and **paths**. Do **not** `read_file`
their deliverables to rewrite them. When the task is done (or the user should see outputs), call
**`render_files`** with **all** final paths — that is how the user reads reports in the terminal.

You may relay **brief** clarifying questions from a specialist to the user when blocked.

## Routing

| Goal | Use |
|------|-----|
| Live web research, citations, long-form markdown from the open web | **`web_agent` tool** — one self-contained brief; specialist uses `tvly` + skills. |
| Read-only SQL on sample ``*.db`` files under `test_demo/dbs/test_dbs` | **`sql_agent` tool** — self-contained brief; every query needs **`db_name`** (e.g. `chinook.db`). |
| PDF workflows | **`pdf_agent` tool**. |
| Hugging Face Hub / MCP | **`hf_agent` tool**; keep delegation brief (if HF is not configured, the tool will say so). |
| Show finished text to the user | **`render_files`** once, with every relevant path — not for mid-task browsing. |

**Delegation:** one strong **`tool` call per specialist** when possible; pass **paths** between
steps, not pasted bodies.
"""


orchestrator_runner = create_deep_agent(
    name="orchestrator",
    description=(
        "Coordinates web research, SQL, PDF, and optional HF workflows via specialist tools. "
        "Uses planning tools; does not execute specialist work itself."
    ),
    subagents=[
        web_agent_runner,
        sql_agent_runner,
        pdf_agent_runner,
        hf_agent_runner,
    ],
    tools=orch_tools,
    skills=[str(ORCH_SKILLS_DIR)],
    system_prompt=DEMO_ORCHESTRATOR_SYS,
    checkpointer=ORCH_DB,
    debug=True,
    backend=DEMO_BACKEND,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepx orchestrator demo (terminal).")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn session with streaming (default if --chat_sync not set).",
    )
    parser.add_argument(
        "--chat_sync",
        action="store_true",
        help="Interactive multi-turn session without token streaming.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log SDK stream events in addition to assistant text (streaming mode only).",
    )
    args, _rest = parser.parse_known_args()

    from deepx_cli.session import run_chat_stream, run_chat_sync

    if not args.chat_sync:
        run_chat_stream(orchestrator_runner, verbose=args.verbose)
    else:
        run_chat_sync(orchestrator_runner)


if __name__ == "__main__":
    main()
