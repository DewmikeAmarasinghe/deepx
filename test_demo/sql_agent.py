"""Single SQLite demo subagent: set ``DB_NAME`` to ``chinook`` or ``northwind`` (see ``TASK_BY_DB``)."""

from __future__ import annotations

from pathlib import Path
from deepx import create_deep_agent
from deepx.backends.protocol import BackendProtocol

from test_demo.sql_tools import create_sql_tools

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
TEST_DBS = DEMO_DIR / "dbs" / "test_dbs"
SQL_SKILLS_DIR = DEMO_DIR / "skills" / "sql"

DB_NAME: str = "chinook"  # set to "northwind" to exercise the other sample DB

NORTHWIND_DB = TEST_DBS / "northwind.db"


def _db_path() -> Path:
    if DB_NAME not in ("chinook", "northwind"):
        raise ValueError(f"Invalid DB_NAME: {DB_NAME!r}")
    return TEST_DBS / f"{DB_NAME}.db"


TASK_BY_DB: dict[str, str] = {
    "chinook": """
I need a trustworthy analytics pack from the Chinook sample database.

First, map the schema enough to explain how tracks, albums, artists, and genres relate.

Then answer: which three genres have the most tracks, and for each of those genres who are the
top three artists by track count? I want the SQL you used, the result tables, and a short
interpretation a product manager could read.

Finally, identify any obvious data-quality caveats (missing links, odd outliers) you noticed
while exploring—no hand-waving, cite what you saw in the data.
""",
    "northwind": """
We are reviewing Northwind-style retail data before a dashboard build.

Produce: (1) each customer's total order count and total quantity ordered across all lines;
(2) average order value per customer; (3) the top five products by revenue (quantity × unit price)
with the SQL shown.

Return ranked tables, note any join assumptions you made, and call out anything that would break
if we moved from this toy schema to a messy production warehouse.
""",
}


def build_sql_agent_runner(*, backend: BackendProtocol, checkpointer: str):
    path = _db_path()
    if not path.is_file():
        return None
    tools = create_sql_tools(str(path), tool_prefix="sql")
    return create_deep_agent(
        name="sql_agent",
        description=(
            f"Read-only SQL over the bundled **{DB_NAME}** SQLite database "
            f"(sql_db_list_tables, sql_db_schema, sql_db_query)."
        ),
        tools=tools,
        skills=[str(SQL_SKILLS_DIR)],
        system_prompt=(
            f"You answer questions about the **{DB_NAME}** sample database only. "
            "Use sql_db_list_tables, sql_db_schema, and sql_db_query. "
            "Read schema-exploration and query-writing skills when the task is non-trivial. "
            "For multi-part asks, call write_todos first, then update_todos as you finish each part. "
            "Return clear tables and explicit SQL."
        ),
        checkpointer=checkpointer,
        backend=backend,
    )


if __name__ == "__main__":
    import os
    import sys
    import uuid

    from deepx.backends.local_shell import LocalShellBackend

    TEST_DBS.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "dbs" / "agent_dbs").mkdir(parents=True, exist_ok=True)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    cp = str(DEMO_DIR / "dbs" / "agent_dbs" / f"sql_agent_{DB_NAME}.db")
    runner = build_sql_agent_runner(
        backend=LocalShellBackend(REPO_ROOT),
        checkpointer=cp,
    )
    if runner is None:
        print(f"Missing database file: {_db_path()}", file=sys.stderr)
        sys.exit(1)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    out = runner.run_sync(TASK_BY_DB[DB_NAME], session_id=sid)
    print(out.output)
