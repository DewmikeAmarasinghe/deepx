"""SQLite demo subagent: set ``SQLITE_DB`` (filename under test_dbs) or ``SQLITE_DB_PATH`` (any .db file)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from deepx import create_deep_agent  # noqa: E402
from deepx.backends.protocol import BackendProtocol  # noqa: E402

from test_demo.sql_tools import create_sql_tools  # noqa: E402

DEMO_DIR = _DEMO_DIR
REPO_ROOT = _REPO_ROOT
TEST_DBS = DEMO_DIR / "dbs" / "test_dbs"
SQL_SKILLS_DIR = DEMO_DIR / "skills" / "sql"


def _db_path() -> Path:
    """Resolve DB: ``SQLITE_DB_PATH`` (absolute file) or ``SQLITE_DB`` (name or path; default ``chinook.db`` under ``test_dbs``)."""
    raw_path = os.environ.get("SQLITE_DB_PATH", "").strip()
    if raw_path:
        p = Path(raw_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SQLITE_DB_PATH is not a file: {p}")
        return p
    raw = os.environ.get("SQLITE_DB", "chinook.db").strip() or "chinook.db"
    p = Path(raw).expanduser()
    if p.is_absolute() or p.parent != Path("."):
        p = p.resolve()
    else:
        p = (TEST_DBS / raw).resolve()
    return p


TASK_BY_STEM: dict[str, str] = {
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

DEFAULT_TASK = """
Open the configured SQLite database: list tables, inspect schema for the main entities, then run
a few exploratory queries that show what the dataset is good for. Return SQL, small result tables,
and a short narrative for a developer who has not seen this file before.
"""


def build_sql_agent_runner(*, backend: BackendProtocol, checkpointer: str):
    try:
        path = _db_path()
    except (FileNotFoundError, OSError):
        return None
    if not path.is_file():
        return None
    tools = create_sql_tools(str(path), tool_prefix="sql")
    return create_deep_agent(
        name="sql_agent",
        description=(
            "Read-only SQL over the configured SQLite database "
            "(sql_db_list_tables, sql_db_schema, sql_db_query)."
        ),
        tools=tools,
        skills=[str(SQL_SKILLS_DIR)],
        system_prompt=(
            "You answer questions about the **configured** SQLite database only. "
            "Use sql_db_list_tables, sql_db_schema, and sql_db_query. "
            "Read schema-exploration and query-writing skills when the task is non-trivial. "
            "For multi-part asks, call write_todos first, then update_todos as you finish each part. "
            "Return clear tables and explicit SQL."
        ),
        checkpointer=checkpointer,
        backend=backend,
    )


def default_task_for_current_db() -> str:
    stem = _db_path().stem
    return TASK_BY_STEM.get(stem, DEFAULT_TASK)


if __name__ == "__main__":
    import uuid

    from deepx.backends.local_shell import LocalShellBackend

    TEST_DBS.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "dbs" / "agent_dbs").mkdir(parents=True, exist_ok=True)

    try:
        db_path = _db_path()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    stem = db_path.stem
    cp = str(DEMO_DIR / "dbs" / "agent_dbs" / f"sql_agent_{stem}.db")
    runner = build_sql_agent_runner(
        backend=LocalShellBackend(REPO_ROOT),
        checkpointer=cp,
    )
    if runner is None:
        print(f"Missing database file: {db_path}", file=sys.stderr)
        sys.exit(1)
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    _script = Path(__file__).resolve()
    _resume = f"{sys.executable} {_script}"
    out = runner.run_sync(default_task_for_current_db(), session_id=sid)
    print(out.output)
    print(f"\nSession: {out.session_id}\nTo resume with this script + same history: `{_resume}` (see script help for session args).\n")
