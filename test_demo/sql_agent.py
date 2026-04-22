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

from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx.factory import DeepAgentRunner, create_deep_agent  # noqa: E402
from test_demo.sql_tools import create_sql_tools  # noqa: E402

DEMO_DIR = _DEMO_DIR
REPO_ROOT = _REPO_ROOT
TEST_DBS = DEMO_DIR / "dbs" / "test_dbs"
SQL_SKILLS_DIR = DEMO_DIR / "skills" / "sql"

demo_backend = FilesystemBackend(REPO_ROOT)
agent_dbs_dir = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
agent_dbs_dir.mkdir(parents=True, exist_ok=True)
sql_session_db = str(agent_dbs_dir / "sql_agent.db")


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


def default_task_for_stem(stem: str) -> str:
    return TASK_BY_STEM.get(stem, DEFAULT_TASK)


has_demo_dbs = TEST_DBS.is_dir() and any(TEST_DBS.glob("*.db"))
sql_tools = create_sql_tools(TEST_DBS, tool_prefix="sql") if has_demo_dbs else []
available_db_names = (
    ", ".join(sorted(p.name for p in TEST_DBS.glob("*.db")))
    if TEST_DBS.is_dir()
    else ""
)

_sql_no_dbs_prompt = """\
You are **sql_agent**, but there are **no** `*.db` files under **test_demo/dbs/test_dbs** in this
workspace (expected: e.g. chinook.db, northwind.db). You have **no** sql_* tools until those files
exist. Reply briefly that SQL demos are unavailable and what the user or orchestrator should add;
do not invent query results.
"""

_sql_active_prompt = f"""\
You answer using **sql_*** tools only against files under **test_demo/dbs/test_dbs**.
Each tool requires **`db_name`**: a filename present there (now: {available_db_names}).
Default demos: **chinook.db** (music retail), **northwind.db** (classic retail) unless the user
picks another listed file.
For non-trivial tasks, follow skills under the sql skill folder.
For multi-part work, use write_todos and refresh the list as steps complete.
Return explicit SQL and readable tables.
"""

sql_agent_runner: DeepAgentRunner = create_deep_agent(
    name="sql_agent",
    description=(
        "Specialist for read-only SQLite on bundled demo databases. "
        "Tools: sql_db_list_tables (discover tables), sql_db_schema (DDL + tiny row samples; "
        "BLOB columns summarized), sql_db_query (SELECT only). "
        "Every call must include db_name (e.g. chinook.db, northwind.db). "
        "For long SQL sessions the orchestrator should call this tool with a full brief. "
        "If no demo DBs are present, the agent reports that state."
    ),
    tools=sql_tools,
    skills=[str(SQL_SKILLS_DIR)],
    system_prompt=_sql_active_prompt if sql_tools else _sql_no_dbs_prompt,
    backend=demo_backend,
    checkpointer=sql_session_db,
    debug=True,
    subagents=None,
)
