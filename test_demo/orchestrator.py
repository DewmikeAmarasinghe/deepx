"""Multi-agent orchestrator demo. Run: ``python test_demo/orchestrator.py`` or ``python -m test_demo.orchestrator``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import RunContextWrapper, function_tool  # noqa: E402

from deepx import create_deep_agent  # noqa: E402
from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.context import AgentContext  # noqa: E402
from deepx_cli import run_interactive  # noqa: E402

from test_demo import sql_agent as sql_mod  # noqa: E402
from test_demo.pdf_agent import pdf_agent_spec  # noqa: E402
from test_demo.sql_tools import create_sql_tools  # noqa: E402
from test_demo.web_agent import web_agent_spec  # noqa: E402


@function_tool
async def render_files(
    ctx: RunContextWrapper[AgentContext],
    paths: list[str],
    max_lines_per_file: int = 400,
) -> str:
    """Render multiple text or markdown files from the session workspace to the host terminal."""
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


DBS_DIR = _REPO_ROOT / "test_demo" / "dbs"
TEST_DBS = DBS_DIR / "test_dbs"
AGENT_DBS = DBS_DIR / "agent_dbs"
for d in (TEST_DBS, AGENT_DBS):
    d.mkdir(parents=True, exist_ok=True)

DEMO_BACKEND = LocalShellBackend(_REPO_ROOT)

ORCH_DB = str(AGENT_DBS / "orchestrator.db")
WEB_DB = str(AGENT_DBS / "web_agent.db")
SQL_AGENT_DB = str(AGENT_DBS / "sql_agent.db")
PDF_AGENT_DB = str(AGENT_DBS / "pdf_agent.db")

_chinook_sql = (
    create_sql_tools(str(TEST_DBS / "chinook.db"), tool_prefix="chinook")
    if (TEST_DBS / "chinook.db").is_file()
    else []
)
_northwind_sql = create_sql_tools(str(TEST_DBS / "northwind.db"), tool_prefix="northwind")

web_agent = web_agent_spec(checkpointer=WEB_DB, interrupt_on=["web_search"])
pdf_agent = pdf_agent_spec(checkpointer=PDF_AGENT_DB)

sql_runner = sql_mod.build_sql_agent_runner(
    backend=DEMO_BACKEND,
    checkpointer=SQL_AGENT_DB,
)

_subagents: list = [web_agent]
if sql_runner is not None:
    _subagents.append(sql_runner)
_subagents.append(pdf_agent)

ORCH_SKILLS_DIR = _REPO_ROOT / "test_demo" / "skills" / "orchestrate"

_orch_tools = [*_chinook_sql, *_northwind_sql, render_files]

agent = create_deep_agent(
    name="orchestrator",
    description=(
        "Coordinates web research, SQL over bundled samples, and PDF workflows. "
        "Uses planning tools; delegates execution to specialists."
    ),
    subagents=_subagents,
    tools=_orch_tools,
    skills=[str(ORCH_SKILLS_DIR)],
    system_prompt=(
        "You are the orchestrator. Use your **orchestrate** skill for workflow norms.\n"
        "For substantial multi-step requests: call **write_todos** before other tools, then "
        "**update_todos** after each major step (including after each subagent returns).\n"
        "Delegate with **self-contained** prompts: **web_agent** for web research and **final "
        "markdown reports** in the session workspace; **sql_agent** for the bundled SQLite sample; "
        "**pdf_agent** for PDF merge/split/extract and related work.\n"
        "Pass **file paths** between steps—never paste large bodies. When the user should see "
        "finished markdown in the terminal, call **render_files** with the artifact path(s).\n"
        "You may answer quick SQL yourself with **chinook_*** / **northwind_*** tools when a "
        "single query is enough; delegate heavier exploration to **sql_agent**."
    ),
    interrupt_on=["web_search"],
    checkpointer=ORCH_DB,
    debug=True,
    backend=DEMO_BACKEND,
)

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
# """
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
# quarter; for each, title, why it matters, and a link. Write paths to a workspace digest and render it.
# """


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--"]
    run_once = "--once" in args
    args = [a for a in args if a != "--once"]

    if run_once:
        import uuid

        sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
        result = agent.run_sync(TASK, session_id=sid)
        print("\n" + "=" * 70)
        print(result.output)
        print("=" * 70)
        print(f"\nSession: {result.session_id}")
    else:
        positional = [a for a in args if not a.startswith("-")]
        resume_sid = positional[0] if positional else os.environ.get("SESSION_ID")
        run_interactive(agent, session_id=resume_sid)
