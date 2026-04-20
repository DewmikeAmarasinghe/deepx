"""SQLite demo subagent: multi-database tools under ``test_demo/dbs/test_dbs``."""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from deepx.backends.protocol import BackendProtocol  # noqa: E402
from deepx.defaults import DEFAULT_MODEL  # noqa: E402
from deepx.factory import create_deep_agent  # noqa: E402
from test_demo.sql_tools import create_sql_tools  # noqa: E402

DEMO_DIR = _DEMO_DIR
REPO_ROOT = _REPO_ROOT
TEST_DBS = DEMO_DIR / "dbs" / "test_dbs"
SQL_SKILLS_DIR = DEMO_DIR / "skills" / "sql"


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


def build_sql_runner(*, backend: BackendProtocol, checkpointer: str, debug: bool = False):
    """Runner for read-only SQL over *.db files in ``test_dbs`` (pass ``db`` on each tool call)."""
    if not TEST_DBS.is_dir():
        return None
    if not any(TEST_DBS.glob("*.db")):
        return None
    tools = create_sql_tools(TEST_DBS, tool_prefix="sql")
    available = ", ".join(sorted(p.name for p in TEST_DBS.glob("*.db"))) or "(none)"
    return create_deep_agent(
        model=DEFAULT_MODEL,
        name="sql_agent",
        description=(
            "Read-only SQL over allowlisted SQLite files in test_dbs "
            "(sql_db_list_tables, sql_db_schema, sql_db_query). "
            "Pass ``db`` (e.g. chinook.db or northwind.db) on every tool call. "
            "Use handoff for long exploratory sessions."
        ),
        tools=tools,
        skills=[str(SQL_SKILLS_DIR)],
        system_prompt=(
            "You answer questions using the **sql_*** tools only against databases under "
            "**test_demo/dbs/test_dbs**. Each tool requires a ``db`` argument: a filename present "
            f"there (available now: {available}).\n"
            "Use **chinook.db** for music retail demos and **northwind.db** for classic retail "
            "scenarios unless the user specifies otherwise.\n"
            "Read schema-exploration and query-writing skills when the task is non-trivial. "
            "For multi-part asks, call write_todos first, then call write_todos again with an updated "
            "full list as you finish each part. "
            "Return clear tables and explicit SQL."
        ),
        backend=backend,
        checkpointer=checkpointer,
        debug=debug,
        include_general_purpose=False,
        subagents=None,
    )


def default_task_for_stem(stem: str) -> str:
    return TASK_BY_STEM.get(stem, DEFAULT_TASK)
