"""SQL specialist subagent: query, design, migrate, and manage databases.

Run standalone::

    python test_demo/sql_agent.py --chat
    python test_demo/sql_agent.py --chat_sync
    python test_demo/sql_agent.py --chat --session <id>
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

_TEST_DBS = _REPO_ROOT / "test_demo" / "dbs" / "test_dbs"
_available_dbs = (
    ", ".join(sorted(p.name for p in _TEST_DBS.glob("*.db")))
    if _TEST_DBS.is_dir()
    else "(none found — run the orchestrator once to create the directory)"
)

sql_sys = f"""\
You are the SQL specialist. You handle all SQL work: writing and running queries, schema
design, migrations, indexing, and performance tuning.

Use **`sqlite3`** on the host for SQLite (always available in this demo). For **PostgreSQL**
or **MySQL**, use `psql` / `mysql` only when those clients exist — see the **sql-toolkit**
skill for patterns.

## Before starting any task

Read the skill files under `/test_demo/skills/sql/` — they document helpers, dialect notes,
and layouts you should follow.

## How to run SQL

You execute SQL through the shell via `execute` — there is no interactive session.
You can also write Python or shell scripts and execute them.

You may also write shell scripts (`.sh`) or Python scripts for more complex workflows —
if user hasn't specified a path, write them to `/_outputs/`, then execute them.

## Demo databases

SQLite files under `/test_demo/dbs/test_dbs/`: {_available_dbs}.
Default to `chinook.db` (music retail) or `northwind.db` (classic ERP) when the user
does not specify a file.

## Output

Always show the SQL and the result together. For aggregations or joins, add one sentence
explaining what the query does.

Choose the output path as follows:
- If the orchestrator specified a path in its brief, write there.
- Otherwise write results or reports to `/_outputs/<descriptive_name>.md` (or `.csv`, `.sql`).

Return only: the output file path(s) and a two-to-three sentence summary. Do not paste
large result sets in the reply — write them to a file and return the path.
"""

sql_agent_runner = create_deep_agent(
    name="sql_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "SQL specialist: write and run queries, design schemas, build migrations, optimise "
        "performance, and manage relational databases (SQLite via `sqlite3`; Postgres/MySQL when "
        "host tools exist). Demo SQLite files "
        f"are at `/test_demo/dbs/test_dbs/` ({_available_dbs}). "
        "Writes reports and query results to `/_outputs/` when requested."
    ),
    tools=None,
    skills=["./test_demo/skills/sql"],
    system_prompt=sql_sys,
    backend=LocalShellBackend(_REPO_ROOT),
    checkpointer=str(_REPO_ROOT / "test_demo" / "dbs" / "agent_dbs" / "sql_agent.db"),
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)