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
demo_backend = FilesystemBackend(REPO_ROOT)
agent_dbs_dir = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
agent_dbs_dir.mkdir(parents=True, exist_ok=True)
sql_session_db = str(agent_dbs_dir / "sql_agent.db")

available_db_names = (
    ", ".join(sorted(p.name for p in TEST_DBS.glob("*.db")))
    if TEST_DBS.is_dir()
    else ""
)

sql_tools = create_sql_tools(TEST_DBS, tool_prefix="sql")

_sql_prompt = f"""\
You answer using **sql_*** tools only against files under **test_demo/dbs/test_dbs**.
Each tool requires **`db_name`**: a filename present there (now: {available_db_names or "(scan folder)"}).
Default demos: **chinook.db** (music retail), **northwind.db** (classic retail) unless the user picks another file.
For non-trivial tasks, follow skills under the sql skill folder.
For multi-part work, use write_todos and refresh the list as steps complete.
Return explicit SQL and readable tables.
When the orchestrator asks for a saved report or digest, write it under **`/_outputs/`** and return that path.
"""

sql_agent_runner: DeepAgentRunner = create_deep_agent(
    name="sql_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "SQLite analyst for **demo databases** under `test_demo/dbs/test_dbs`. "
        "chinook.db and northwind.db are configured there."
        "Tools: **sql_db_list_tables**, **sql_db_schema** (DDL + small samples; BLOBs summarized), "
        "**sql_db_query** (SELECT-only). **Every** tool call needs **db_name** (e.g. `chinook.db`, "
        "`northwind.db`). Best for aggregations, joins, and explaining schema; returns SQL + readable "
        "tables. If no `*.db` files exist, the runner is a no-op stub—say so clearly."
        "Writes outputs to agreed paths (typically **/_outputs/**). and returns paths plus a short summary."
    ),
    tools=sql_tools,
    skills=["./test_demo/skills/sql"],
    system_prompt=_sql_prompt,
    backend=demo_backend,
    checkpointer=sql_session_db,
    debug=True,
    subagents=None,
)
