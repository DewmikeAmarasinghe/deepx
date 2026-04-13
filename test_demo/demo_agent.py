"""Multi-agent orchestrator demo.

- Web research: `web_agent` tool (subagent)
- Writing: `writer` tool (subagent)
- Text-to-SQL: `sql_chinook` and `sql_northwind` (separate SQLite tool prefixes)
- PDF/forms: `pdf_agent` (subagent with bundled ``test_demo/skills/pdf``)

Usage:
    python test_demo/demo_agent.py              # interactive (default)
    python test_demo/demo_agent.py <session_id> # resume chat
    python test_demo/demo_agent.py --once       # one-shot TASK

Type /bye to exit (handled by the CLI, not sent to the model).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deepx import create_deep_agent  # noqa: E402
from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx_cli import run_interactive  # noqa: E402

from test_demo.tools import (  # noqa: E402
    CHINOOK_DB,
    DBS_DIR,
    NORTHWIND_DB,
    PDF_SKILLS_DIR,
    SKILLS_DIR,
    SQL_SKILLS_DIR,
    create_sql_tools,
    render_file,
    web_extract,
    web_map,
    web_search,
)

DBS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_BACKEND = FilesystemBackend(_REPO_ROOT)

ORCH_DB = str(DBS_DIR / "orchestrator.db")
WEB_DB = str(DBS_DIR / "web_agent.db")
WRITER_DB = str(DBS_DIR / "writer.db")
SQL_CHINOOK_DB = str(DBS_DIR / "sql_chinook.db")
SQL_NORTHWIND_DB = str(DBS_DIR / "sql_northwind.db")
PDF_AGENT_DB = str(DBS_DIR / "pdf_agent.db")

web_agent = {
    "name": "web_agent",
    "description": (
        "Web research: give the full topic list in one call. Uses Tavily-backed tools; "
        "writes notes in the session workspace under a sensible folder, returns file paths."
    ),
    "system_prompt": (
        "You are a web research specialist. Your tools: web_search, web_extract, web_map.\n"
        "You receive a list of research topics in a single call. Research ALL of them, then "
        "consolidate findings into **as few markdown files as practical** (often one primary "
        "file plus optional short sidecars only if separation clearly helps the orchestrator).\n"
        "Save artifacts in the session workspace (for example a `research/` folder); the "
        "orchestrator shares that tree with you.\n"
        "Structure: clear headings per topic, key facts in bullets, inline citations [1][2], "
        "sources section at the bottom. Do NOT paste raw tool output.\n"
        "Return the paths of the files you created so the orchestrator can pass them on."
    ),
    "tools": [web_search, web_extract, web_map],
    "skills": [str(SKILLS_DIR), str(SKILLS_DIR / "arxiv-search")],
    "interrupt_on": ["web_search"],
    "checkpointer": WEB_DB,
}

writer = {
    "name": "writer",
    "description": (
        "Technical writer. Give it a task description and one or more research file paths. "
        "It reads the files and writes the polished document to disk, then returns that path."
    ),
    "system_prompt": (
        "You are a professional technical writer. Read the research paths you are given with "
        "`read_file`, then produce the final markdown using `write_file` in the session workspace. "
        "Choose a clear folder and filename that match the brief. Return the artifact path(s) "
        "you wrote so the orchestrator can call `render_file`."
    ),
    "checkpointer": WRITER_DB,
}

pdf_agent_spec = {
    "name": "pdf_agent",
    "description": (
        "PDF and form workflows: filling PDFs, extracting form structure, validation images. "
        "Uses the bundled pdf skill; may run shell commands via `execute` when scripts are needed."
    ),
    "system_prompt": (
        "You specialise in PDF and form tasks. Read the skill entry paths listed in your "
        "instructions first (`read_file` on each relevant `SKILL.md`), then follow the "
        "workflows there. Use `execute` for bundled Python scripts when appropriate; keep "
        "outputs in the session workspace and return paths to anything the orchestrator should read."
    ),
    "skills": [str(PDF_SKILLS_DIR)],
    "checkpointer": PDF_AGENT_DB,
}

_chinook_sql = (
    create_sql_tools(str(CHINOOK_DB), tool_prefix="chinook") if CHINOOK_DB.exists() else []
)
_northwind_sql = create_sql_tools(str(NORTHWIND_DB), tool_prefix="northwind")

sql_chinook_runner = (
    create_deep_agent(
        model="gpt-5-nano",
        name="sql_chinook",
        description=(
            "Answers questions about the **Chinook** SQLite database using read-only "
            "chinook_sql_* tools."
        ),
        tools=_chinook_sql,
        skills=[str(SQL_SKILLS_DIR)],
        system_prompt=(
            "You are the Chinook database specialist. Use chinook_sql_db_list_tables, "
            "chinook_sql_db_schema, and chinook_sql_db_query only. "
            "Read the schema-exploration and query-writing skills when needed. Return clear answers."
        ),
        checkpointer=SQL_CHINOOK_DB,
        backend=DEMO_BACKEND,
    )
    if _chinook_sql
    else None
)

sql_northwind_runner = create_deep_agent(
    model="gpt-5-nano",
    name="sql_northwind",
    description=(
        "Answers questions about the **Northwind** SQLite database using read-only "
        "northwind_sql_* tools."
    ),
    tools=_northwind_sql,
    skills=[str(SQL_SKILLS_DIR)],
    system_prompt=(
        "You are the Northwind database specialist. Use northwind_sql_db_list_tables, "
        "northwind_sql_db_schema, and northwind_sql_db_query only. "
        "Read the schema-exploration and query-writing skills when needed. Return clear answers."
    ),
    checkpointer=SQL_NORTHWIND_DB,
    backend=DEMO_BACKEND,
)

_subagents: list = [web_agent, writer]
if sql_chinook_runner is not None:
    _subagents.append(sql_chinook_runner)
_subagents.append(sql_northwind_runner)
_subagents.append(pdf_agent_spec)

_orch_tools = [*_chinook_sql, *_northwind_sql, render_file]

agent = create_deep_agent(
    model="gpt-5-nano",
    name="orchestrator",
    description=(
        "General-purpose orchestrator. Handles web research, document writing, "
        "text-to-SQL against Chinook and/or Northwind, and PDF/form workflows via pdf_agent."
    ),
    subagents=_subagents,
    tools=_orch_tools,
    skills=[str(SKILLS_DIR)],
    system_prompt=(
        "You are the orchestrator. For web research plus reports: call the `web_agent` tool, "
        "then the `writer` tool. For database questions: call `sql_chinook` for the Chinook "
        "music store schema or `sql_northwind` for the Northwind retail sample—pick the tool "
        "that matches the user's database. For PDFs, forms, or fillable-field workflows, call "
        "`pdf_agent`. "
        "Pass file paths between steps; do not paste large file bodies into prompts. "
        "When the writer returns a final document path, call `render_file` on that path so the "
        "user sees it in the terminal."
    ),
    interrupt_on=["web_search"],
    checkpointer=ORCH_DB,
    debug=True,
    backend=DEMO_BACKEND,
)

# Default one-shot task (switch by uncommenting one block).
TASK = """
Investigate the long-term viability of sodium-ion batteries as a sustainable alternative to
lithium-ion technology in the electric vehicle market. Analyse current energy density limitations,
the geopolitical stability of the raw material supply chain, and existing manufacturing hurdles
for large-scale adoption. Identify key startups leading the sector and compare the lifecycle
environmental impact of sodium versus lithium extraction.
After investigating, write a comprehensive report.
"""

# TASK_DEEP_RESEARCH = """
# Multi-source deep research: pick a technical controversy in AI safety, gather contrasting
# positions with citations, and have the writer produce a balanced memo in the session workspace.
# """

# TASK_CHINOOK_SQL = """
# Using sql_chinook: list tables, then answer which genre has the most tracks and show the top
# three artists by total track count (read-only SQL).
# """

# TASK_NORTHWIND_SQL = """
# Using sql_northwind: list tables, then write a query that joins customers to orders and returns
# each customer's total order count (read-only SQL).
# """

# TASK_PDF = """
# Using pdf_agent: read the pdf skill, then outline how you would validate a fillable PDF form
# using the bundled scripts (no need to run if dependencies are missing—describe the steps).
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
