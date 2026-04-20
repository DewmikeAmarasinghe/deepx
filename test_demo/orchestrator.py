"""Multi-agent orchestrator demo.

Run from the repository root (the ``test_demo`` tree is not shipped in the wheel)::

    uv sync --extra demo
    uv run --extra demo python -m test_demo.orchestrator --chat
    uv run --extra demo python -m test_demo.orchestrator --once

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

from deepx import SubagentRef, create_deep_agent  # noqa: E402
from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.context import AgentContext  # noqa: E402
from deepx.defaults import DEFAULT_MODEL  # noqa: E402
from deepx.system_prompt import COORDINATOR_ROLE_PROMPT  # noqa: E402
from test_demo import sql_agent as sql_mod  # noqa: E402
from test_demo.pdf_agent import build_pdf_runner  # noqa: E402
from test_demo.web_agent import build_web_runner  # noqa: E402


@function_tool
async def render_files(
    ctx: RunContextWrapper[AgentContext],
    paths: list[str],
    max_lines_per_file: int = 400,
) -> str:
    """Render multiple text or markdown files from the project tree to the host terminal."""
    sid = ctx.context.session_id
    lim = max(1, min(max_lines_per_file, 5000))
    parts: list[str] = []
    for path in paths:
        p = (path or "").strip()
        if not p:
            continue
        rr = ctx.context.backend.read(sid, p, 0, lim)
        bar = "=" * 72
        if rr.error:
            print(f"\n{bar}\n{p}\n{rr.error}\n{bar}\n", flush=True)
            parts.append(f"{p}: error")
            continue
        text = rr.content or ""
        print(f"\n{bar}\n{p}\n{bar}\n{text}\n{bar}\n", flush=True)
        parts.append(f"{p}: {len(text)} chars")
    return "Rendered: " + "; ".join(parts) if parts else "No paths provided."


DBS_DIR = REPO_ROOT / "test_demo" / "dbs"
TEST_DBS = DBS_DIR / "test_dbs"
AGENT_DBS = DBS_DIR / "agent_dbs"
for d in (TEST_DBS, AGENT_DBS):
    d.mkdir(parents=True, exist_ok=True)

DEMO_BACKEND = LocalShellBackend(REPO_ROOT)

ORCH_DB = str(AGENT_DBS / "orchestrator.db")
WEB_DB = str(AGENT_DBS / "web_agent.db")
SQL_AGENT_DB = str(AGENT_DBS / "sql_agent.db")
PDF_AGENT_DB = str(AGENT_DBS / "pdf_agent.db")

subagents: list[SubagentRef] = [
    SubagentRef(build_web_runner(backend=DEMO_BACKEND, checkpointer=WEB_DB, debug=True)),
]
_sql = sql_mod.build_sql_runner(backend=DEMO_BACKEND, checkpointer=SQL_AGENT_DB, debug=True)
if _sql is not None:
    subagents.append(SubagentRef(_sql, expose="handoff"))
subagents.append(
    SubagentRef(build_pdf_runner(backend=DEMO_BACKEND, checkpointer=PDF_AGENT_DB, debug=True))
)

ORCH_SKILLS_DIR = REPO_ROOT / "test_demo" / "skills" / "orchestrate"
SKILL_CREATOR_DIR = REPO_ROOT / "test_demo" / "skills" / "skill-creator"
orch_tools = [render_files]


def build_orchestrator_runner():
    return create_deep_agent(
        model=DEFAULT_MODEL,
        name="orchestrator",
        description=(
            "Coordinates web research, SQL (via sql_agent), and PDF workflows. "
            "Uses planning tools; delegates execution to specialists."
        ),
        subagents=subagents,
        tools=orch_tools,
        skills=[str(ORCH_SKILLS_DIR), str(SKILL_CREATOR_DIR)],
        system_prompt=(
            f"{COORDINATOR_ROLE_PROMPT}\n"
            "\n## Demo specialists\n\n"
            "Use **web_agent** (tool) for web research and written deliverables—**self-contained** "
            "prompts with sources and artifact paths.\n"
            "For SQL over allowlisted files in **test_demo/dbs/test_dbs**, use the "
            "**`transfer_to_sql_agent`** handoff. The specialist's tools require a ``db`` argument "
            "(e.g. **chinook.db** or **northwind.db**) on every call.\n"
            "Use **pdf_agent** (tool) for PDF merge/split/extract and related work.\n"
            "When the user should see finished markdown in the terminal, call **render_files** with "
            "the artifact path(s). You may use **execute** for host commands when appropriate."
        ),
        checkpointer=ORCH_DB,
        debug=True,
        backend=DEMO_BACKEND,
    )


agent = build_orchestrator_runner()

TASK = """
I want a clear, well-sourced picture of sodium-ion versus lithium-ion for electric vehicles—
energy density limits, materials and geopolitics, manufacturing scale-up, who is leading,
and lifecycle environmental trade-offs—then a single strong markdown report in the workspace
I can open, and show me that report in the terminal when it is ready.
"""
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
        help="Interactive multi-turn session (default if neither flag is set).",
    )
    args, rest = parser.parse_known_args()

    from deepx_cli.session import run_chat, run_once

    runner = build_orchestrator_runner()
    task = " ".join(rest).strip() or TASK

    if args.once:
        run_once(runner, task)
    else:
        run_chat(runner)


if __name__ == "__main__":
    main()
