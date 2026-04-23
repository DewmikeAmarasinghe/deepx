"""Multi-agent orchestrator demo.

Run from the repository root (the ``test_demo`` tree is not shipped in the wheel)::

    uv sync --extra demo
    python -m test_demo.orchestrator --chat
    python -m test_demo.orchestrator --chat_sync

``--chat`` (default) streams assistant tokens to the terminal.

Multi-line paste in the CLI: type ``\"\"\"`` on the first line, paste your message, then ``\"\"\"`` and Enter.

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

**Coordinate specialists when the task matches the routing table below.** For those tasks you do **not**
run web/Tavily, SQL, PDF, or HF work yourself — you delegate with a **single, self-contained brief**
(including output paths, `db_name` when needed, and reasonable defaults like “top 3 with ties” if the user
already implied them).

**General work (no matching specialist):** answer from context, use your file tools, or guide the user.
Examples: skill/marketplace setup, LobeHub install steps, chit-chat, or anything that is **not** purely
web research, SQL on demo DBs, PDF workflows, or HF Hub MCP. This demo orchestrator has **no** host `execute`
tool — you cannot run shell commands yourself; use `read_file` / planning tools and clear instructions, or
route web execution to **`web_agent`** when the user needs live fetching.

**Delegate first; almost never ask the user to “confirm” the task shape.** If the user already stated
constraints (e.g. “top three genres and top three artists”), **do not** re-ask or paraphrase into clarifying
questions — pick sound defaults, state them in the brief, and call the specialist. Ask the user **only** when
something **objectively required** is missing (e.g. which `*.db` when several apply and the user gave none).

**Specialist outputs:** In every delegation brief, tell the specialist to **write deliverables to files**
under **`/_outputs/`** (exact paths you agree on, e.g. `/_outputs/diffusion_models_summary.md`) and to
return **only paths plus a short summary** — not full file bodies.

**After a specialist tool returns:** trust its **paths** and **short summary**. Do **not** `read_file` those
files. When outputs should be visible in the terminal, call **`render_files`** once with **all** final paths.

**If a specialist’s reply contains a question for the user:** relay it briefly; do not stack your own
extra questions on top.

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
        "Coordinates web research, SQL, PDF, and optional HF workflows via specialist tools when applicable; "
        "handles general tasks and file-based work itself when no specialist fits. Uses planning tools."
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
    args, _rest = parser.parse_known_args()

    from deepx_cli.session import run_chat_stream, run_chat_sync

    if not args.chat_sync:
        run_chat_stream(orchestrator_runner)
    else:
        run_chat_sync(orchestrator_runner)


if __name__ == "__main__":
    main()
