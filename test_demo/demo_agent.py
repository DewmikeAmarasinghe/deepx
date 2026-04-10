"""Multi-agent orchestrator demo.

- Web research: web_agent_subagent
- Writing: writer_subagent
- Text-to-SQL (Chinook): sql_agent_subagent (SQL tools only on that subagent)

Usage:
    python test_demo/demo_agent.py              # interactive (default)
    python test_demo/demo_agent.py <session_id> # resume chat
    python test_demo/demo_agent.py --once       # one-shot TASK

Type /bye to exit (handled by the CLI, not sent to the model).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx import create_deep_agent
from deepx_cli import run_interactive

load_dotenv()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILLS_ROOT = os.path.join(_DEMO_DIR, "skills")
_SQL_SKILLS = os.path.join(_SKILLS_ROOT, "sql")
_CHINOOK_DB = os.path.join(_DEMO_DIR, "chinook.db")


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------
@function_tool
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """
    Perform an AI-optimized web search using the Tavily API.
    
    Use this tool to find real-time information, news, or URLs related to a specific query.
    It returns a list of search results including titles, source URLs, and brief content snippets.
    
    Args:
        query: The natural language search query or keywords.
    """
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot perform web search."
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": 5},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[{i + 1}] {res['title']}\nURL: {res['url']}\n{res['content']}"
        for i, res in enumerate(results)
    )


@function_tool
async def web_extract(ctx: RunContextWrapper, urls: list[str]) -> str:
    """Extract page text for one or more URLs. Returns JSON: [{"url": "...", "content": "..."}, ...]."""
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot extract content."
    if not urls:
        return "[]"
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/extract",
            json={"api_key": TAVILY_KEY, "urls": urls},
            timeout=60,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    out: list[dict[str, str]] = []
    for item in results:
        out.append(
            {
                "url": str(item.get("url", "")),
                "content": str(item.get("raw_content", item.get("content", ""))),
            }
        )
    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# SQL tools
# ---------------------------------------------------------------------------
def create_sql_tools(sqlite_path: str) -> list:
    """Read-only SQL tools for a SQLite file."""

    @function_tool
    def sql_db_list_tables(ctx: RunContextWrapper) -> str:
        """List all tables in the database."""
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return ", ".join(r[0] for r in rows) or "No tables found."

    @function_tool
    def sql_db_schema(ctx: RunContextWrapper, table_names: str) -> str:
        """CREATE TABLE plus sample rows for comma-separated table names."""
        names = [n.strip() for n in table_names.split(",") if n.strip()]
        parts: list[str] = []
        with sqlite3.connect(sqlite_path) as conn:
            for name in names:
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
                if not row:
                    parts.append(f"Table '{name}' not found.")
                    continue
                schema = row[0]
                sample_rows = conn.execute(f"SELECT * FROM {name} LIMIT 3").fetchall()
                cols = [
                    d[0]
                    for d in conn.execute(f"SELECT * FROM {name} LIMIT 0").description
                ]
                sample = "\n".join(str(dict(zip(cols, r))) for r in sample_rows)
                parts.append(f"-- {name}\n{schema}\n\n-- Sample rows:\n{sample}")
        return "\n\n".join(parts)

    @function_tool
    def sql_db_query(ctx: RunContextWrapper, query: str) -> str:
        """Run a read-only SELECT. No INSERT/UPDATE/DELETE/DROP. Use LIMIT if needed."""
        q = query.strip()
        upper = q.upper()
        for forbidden in (
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "REPLACE",
        ):
            if forbidden in upper:
                return f"Error: {forbidden} not allowed. Use SELECT only."
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(q).fetchall()
        if not rows:
            return "Query returned no rows."
        keys = list(rows[0].keys())
        lines = [" | ".join(keys), "-" * max(20, len(" | ".join(keys)))]
        for row in rows:
            lines.append(" | ".join(str(row[k]) for k in keys))
        return "\n".join(lines)

    return [sql_db_list_tables, sql_db_schema, sql_db_query]


# ---------------------------------------------------------------------------
# Subagent specs
# ---------------------------------------------------------------------------
web_agent = {
    "name": "web_agent",
    "description": (
        "Web research: give the full topic list in one call. Searches and extracts content, "
        "writes notes under /research/, returns file paths."
    ),
    "system_prompt": (
        "You are a web research specialist. Your tools: web_search, web_extract.\n"
        "You receive a list of research topics in a single call. Research ALL of them, then "
        "consolidate every finding into ONE markdown file at research/findings.md.\n"
        "Structure: clear headings per topic, key facts in bullets, inline citations [1][2], "
        "sources section at the bottom. Do NOT paste raw tool output.\n"
        "Return exactly: 'Saved research to: research/findings.md'"
    ),
    "tools": [web_search, web_extract],
    "interrupt_on": ["web_search"],
    "checkpointer": "test_demo/demo_agent.db"
}

writer = {
    "name": "writer",
    "description": (
        "Technical writer. Give it a task description and one or more research file paths. "
        "It reads the files and returns the full polished document inline."
    ),
    "system_prompt": (
        "You are a professional technical writer. "
        "Read the research files you are given with read_file, then write the requested document. "
        "Return the complete document inline — do not save to a file."
    ),
    "checkpointer": "test_demo/demo_agent.db"
}

_chinook_db = os.path.join(_DEMO_DIR, "chinook.db")
_sql_tools = create_sql_tools(_chinook_db) if os.path.exists(_chinook_db) else []

sql_agent_runner = create_deep_agent(
    model="gpt-5-nano",
    name="sql_agent",
    description="Orchestrates web research, writing, and SQL via specialised subagents.",
    tools=_sql_tools,
    skills=[_SQL_SKILLS],
    system_prompt=(
        "You are the database specialist. Use sql_db_list_tables, sql_db_schema, and sql_db_query. "
        "Read the schema-exploration and query-writing skills when needed. Return clear answers."
    ),
    checkpointer="test_demo/demo_agent.db"
)

# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


agent = create_deep_agent(
    model="gpt-5-nano",
    name="orchestrator",
    description=(
        "General-purpose orchestrator. Handles web research, document writing, "
        "and text-to-SQL queries against the Chinook database."
    ),
    subagents=[web_agent, writer],
    tools=_sql_tools,
    skills=[_DEMO_DIR],
    system_prompt=(
        "You are the orchestrator. For web research + reports: delegate to web_agent_subagent then "
        "writer_subagent. For database questions: delegate to sql_agent_subagent. "
        "Pass file paths between steps; do not paste large file bodies into prompts."
    ),
    interrupt_on=["web_search"],
    checkpointer="test_demo/demo_agent.db",
    debug=True,
)

TASK = """
Investigate the long-term viability of sodium-ion batteries as a sustainable alternative to
lithium-ion technology in the electric vehicle market. Analyse current energy density limitations,
the geopolitical stability of the raw material supply chain, and existing manufacturing hurdles
for large-scale adoption. Identify key startups leading the sector and compare the lifecycle
environmental impact of sodium versus lithium extraction.
After investigating, write a comprehensive report.
"""

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
        run_interactive(agent, session_id=session_id)

