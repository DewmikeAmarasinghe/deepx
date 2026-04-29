"""SQLite demo subagent: host ``sqlite3`` CLI over ``test_demo/dbs/test_dbs`` (no custom SQL tools)."""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.factory import DeepAgentRunner, create_deep_agent  # noqa: E402

DEMO_DIR = _DEMO_DIR
REPO_ROOT = _REPO_ROOT
TEST_DBS = DEMO_DIR / "dbs" / "test_dbs"
demo_backend = LocalShellBackend(REPO_ROOT)
agent_dbs_dir = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
agent_dbs_dir.mkdir(parents=True, exist_ok=True)
sql_session_db = str(agent_dbs_dir / "sql_agent.db")

available_db_names = (
    ", ".join(sorted(p.name for p in TEST_DBS.glob("*.db")))
    if TEST_DBS.is_dir()
    else ""
)

_sql_prompt = f"""\
You analyse the demo SQLite files under **test_demo/dbs/test_dbs** using the **host** (see skills).

**Execution:** use the built-in **`execute`** tool from the **repository root** (project root for this run).
Run **`sqlite3`** as a single non-interactive command per call, e.g.:
- `sqlite3 test_demo/dbs/test_dbs/chinook.db ".tables"`
- `sqlite3 test_demo/dbs/test_dbs/northwind.db ".schema Tablename"`
- `sqlite3 test_demo/dbs/test_dbs/chinook.db "SELECT ... LIMIT 50;"`

Databases present now: **{available_db_names or "(scan test_demo/dbs/test_dbs)"}**. Defaults: **chinook.db**, **northwind.db** unless the user names another file there.

**Skills:** **`read_file`** on **sql-assistant**, **sql-query-generator**, and **sql-toolkit** SKILL paths before non-trivial SQL.
Use **`read_file`** / **`grep`** for exploration; prefer read-only **`SELECT`**; do not mutate user DBs unless the user explicitly asks.
For multi-part work: **`write_todos`** and refresh as steps complete.

Return the SQL you relied on and readable result summaries.
"""

sql_agent_runner: DeepAgentRunner = create_deep_agent(
    name="sql_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "SQLite analyst for **demo databases** under `test_demo/dbs/test_dbs`. "
        "Uses **`execute`** with the host **`sqlite3`** CLI (no in-process db_* tools). "
        "chinook.db and northwind.db are typical samples. "
        "Follow **sql-assistant** / **sql-query-generator** / **sql-toolkit** skills for methodology. "
        "Returns paths plus a short summary when writing files."
    ),
    tools=[],
    skills=[
        "./test_demo/skills/sql/sql-assistant",
        "./test_demo/skills/sql/sql-query-generator",
        "./test_demo/skills/sql/sql-toolkit",
    ],
    system_prompt=_sql_prompt,
    backend=demo_backend,
    checkpointer=sql_session_db,
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)


def main() -> None:
    from deepx_cli.cli import run_interactive_cli

    run_interactive_cli(
        sql_agent_runner,
        description="Deepx SQL specialist demo (terminal).",
    )


if __name__ == "__main__":
    main()
