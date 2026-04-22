"""Multi-agent orchestrator demo.

Run from the repository root (the ``test_demo`` tree is not shipped in the wheel)::

    uv sync --extra demo
    python -m test_demo.orchestrator --chat
    python -m test_demo.orchestrator --chat_sync
    python -m test_demo.orchestrator --once
    python -m test_demo.orchestrator --once --verbose

``--chat`` (default) streams assistant tokens; ``--verbose`` adds SDK stream events (agent
switches, tools, raw events). ``--chat_sync`` runs one full model turn per message without token
streaming. ``--once`` runs a single task (default in-module ``TASK`` or extra CLI args) and exits.

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

from deepx import SubagentRef, create_deep_agent  # noqa: E402
from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
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

DEMO_BACKEND = LocalShellBackend(REPO_ROOT)

ORCH_DB = str(AGENT_DBS / "orchestrator.db")

ORCH_SKILLS_DIR = REPO_ROOT / "test_demo" / "skills" / "orchestrate"
orch_tools = [render_files]

DEMO_ORCHESTRATOR_SYS = """\
## Role

You **only coordinate** specialists. You do **not** do deep web research, SQL, or PDF work yourself,
and you do **not** ingest full specialist file bodies for your own answers — you use **paths**,
**short** specialist summaries, and **`render_files`** when the user should see finished text.

You may relay **brief** clarifying questions from a specialist to the user when blocked.

## Routing

| Goal | Use |
|------|-----|
| Live web research, citations, long-form markdown from the open web | **Hand off** with **`transfer_to_web_agent`** (same run/session). Give one self-contained brief. |
| Read-only SQL on sample ``*.db`` files under `test_demo/dbs/test_dbs` | **Call the `sql_agent` tool** with a self-contained brief. Every SQL call needs **`db_name`** (e.g. `chinook.db`, `northwind.db`). |
| PDF merge/split/extract, forms, pdf skill workflows | **Call the `pdf_agent` tool** with a self-contained brief. |
| Hugging Face Hub / docs via MCP (only if this run includes `hf_agent`) | **Call the `hf_agent` tool**; keep answers short; large dumps under `/_outputs/`. |
| Show the user finished markdown or text in the terminal | **`render_files`** once task is done — all relevant final paths (not for exploratory reads mid-delegation). |
| Host shell when file tools are not enough | **`execute`** (bounded; same backend as the orchestrator). |

**Delegation:** one strong call or handoff per specialist when possible; pass **file paths** between
steps, not huge pasted bodies. Specialists return paths; you **`render_files`** for the user at the end.
"""


orchestrator_runner = create_deep_agent(
    name="orchestrator",
    description=(
        "Coordinates web research (handoff), SQL (tool), and PDF workflows. "
        "Uses planning tools; delegates execution to specialists."
    ),
    subagents=[
        SubagentRef(web_agent_runner, expose="handoff"),
        *(
            [SubagentRef(sql_agent_runner, expose="tool")]
            if sql_agent_runner is not None
            else []
        ),
        SubagentRef(pdf_agent_runner),
        *([SubagentRef(hf_agent_runner)] if hf_agent_runner is not None else []),
    ],
    tools=orch_tools,
    skills=[str(ORCH_SKILLS_DIR)],
    system_prompt=DEMO_ORCHESTRATOR_SYS,
    checkpointer=ORCH_DB,
    debug=True,
    backend=DEMO_BACKEND,
)

TASK = """
I want a clear, well-sourced picture of sodium-ion versus lithium-ion for electric vehicles—
energy density limits, materials and geopolitics, manufacturing scale-up, then write a single strong markdown report in the workspace
I can open, and show me that report in the terminal when it is ready.
"""
# TASK = """
# Search Hugging Face Hub via MCP (hf_agent): shortlist three recent diffusion-model papers with
# links and one-line relevance; save a workspace digest path and summarise trade-offs for practitioners.
# Requires HF_TOKEN and Node/npx for the MCP server.
# """
# TASK = """
# A balanced deep-dive memo on a contested topic in AI safety: competing claims, evidence on both
# sides, and a concise set of open questions — written so a technical reader can act on it.
# End with a workspace markdown path and render it.
# """
# TASK = """
# From the Chinook sample DB: which three genres have the most tracks, and within each genre who
# are the top three artists by track count? Show SQL and tables; then a short executive readout.
# # """
# TASK = """
# Northwind-style retail: each customer's order count, total quantity, and average order value,
# ranked. I need SQL, results, and one paragraph on what would break in a messy production schema.
# """
# TASK = """
# I have two research PDFs under /test_demo/pdfs/ (attention.pdf and gpt4.pdf). Summarize each
# in a structured way (ideas, architectures, limitations), compare how themes evolved, extract any
# key tables or numbers you can, then produce one combined workspace report and a merged PDF—
# return paths and render the report.
# """
# TASK = """
# arXiv: shortlist recent papers that materially change efficient LLM inference assumptions this
# quarter; for each, title, why it matters, and a link. Write paths to a workspace digest and
# render it.
# """
# Smoke-test **execute** (host shell, cwd = repo root): uncomment to verify LocalShellBackend.
# TASK = """
# Run a short non-interactive shell check: `uname -a` and `python3 -c "import sys; print(sys.
# version)"`.
# Paste both outputs in your reply and say whether the shell looks healthy.
# """
# TASK = """
# Curl https://lobehub.com/skills/openclaw-openclaw-weather/skill.md, then follow the instructions
# to set up LobeHub Skills Marketplace and install the skill. Once installed, read the SKILL.md file
# in the installed directory and follow its instructions to complete the task.
# """


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepx orchestrator demo (terminal).")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the default TASK (or pass a prompt as extra args) once and exit.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn session with streaming (default if neither mode flag is set).",
    )
    parser.add_argument(
        "--chat_sync",
        action="store_true",
        help="Interactive multi-turn session without token streaming.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log SDK stream events (agents, tools, raw) in addition to assistant text.",
    )
    args, rest = parser.parse_known_args()

    from deepx_cli.session import run_chat_stream, run_chat_sync, run_once

    task = " ".join(rest).strip() or TASK

    if args.once:
        run_once(orchestrator_runner, task, verbose=args.verbose)
    elif args.chat_sync:
        run_chat_sync(orchestrator_runner)
    else:
        run_chat_stream(orchestrator_runner, verbose=args.verbose)


if __name__ == "__main__":
    main()
