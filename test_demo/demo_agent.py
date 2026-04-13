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

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx import create_deep_agent
from deepx.backends.filesystem import FilesystemBackend
from deepx.context import AgentContext
from deepx_cli import run_interactive

load_dotenv()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILLS_ROOT = os.path.join(_DEMO_DIR, "skills")
_SQL_SKILLS = os.path.join(_SKILLS_ROOT, "sql")
_CHINOOK_DB = os.path.join(_DEMO_DIR, "chinook.db")
_DEMO_BACKEND = FilesystemBackend(Path(".deepx").resolve())


@function_tool
async def web_search(ctx: RunContextWrapper, queries: list[str]) -> str:
    """
    Run one or more independent search requests against the public web index.

    Each query is executed in parallel. Use a separate query string when the information
    needed is unrelated; combine related facets into one query to reduce cost.

    Returns JSON: a list of objects with "query", "answer", "results", and other fields
    returned by the search API for that query.
    """
    _ = ctx
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot perform web search."
    qs = [q.strip() for q in queries if q and str(q).strip()]
    if not qs:
        return "No queries provided."
    async with httpx.AsyncClient() as client:

        async def one(q: str) -> dict:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": q,
                    "max_results": 5,
                },
                timeout=25,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "query": q,
                "answer": data.get("answer"),
                "results": data.get("results", []),
                "images": data.get("images", []),
            }

        rows = await asyncio.gather(*[one(q) for q in qs])
    return json.dumps(rows, indent=2)


@function_tool
async def web_extract(
    ctx: RunContextWrapper,
    urls: list[str],
    query: str | None = None,
    chunks_per_source: int | None = None,
) -> str:
    """
    Extract and clean full webpage content from specific URLs using a hosted extraction service.

    Acts as a structured fetch that returns core text. Use when you already have URLs and
    need their text for analysis or summarization.

    Args:
        urls: One or more valid HTTP(S) URLs to extract.
        query: Optional intent string; when set, the service may rerank content chunks.
        chunks_per_source: When ``query`` is set, maximum chunks per URL (1 to 5).
    """
    _ = ctx
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot extract content."
    if not urls:
        return "[]"
    body: dict = {"api_key": TAVILY_KEY, "urls": urls}
    if query:
        body["query"] = query
    if chunks_per_source is not None:
        body["chunks_per_source"] = max(1, min(5, int(chunks_per_source)))
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/extract",
            json=body,
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
    results = data.get("results", [])
    failed = data.get("failed_results", [])
    out = {"results": results, "failed_results": failed}
    return json.dumps(out, indent=2)


@function_tool
async def web_map(
    ctx: RunContextWrapper,
    url: str,
    instructions: str | None = None,
    max_depth: int = 1,
    max_breadth: int = 12,
    limit: int = 40,
) -> str:
    """
    Discover URLs reachable from a single root URL on one site (breadth-first link crawl).

    Use after you already know which host to explore. Prefer narrow limits to control cost.
    This issues one map request per call (no internal parallel map fan-out).

    Args:
        url: Root page where crawling starts (scheme required, e.g. https://example.com).
        instructions: Optional natural-language focus for the crawl (may increase cost).
        max_depth: How many link levels to follow (1 to 5).
        max_breadth: Maximum links to follow from each page (1 to 500).
        limit: Maximum URLs to return before stopping (>= 1).
    """
    _ = ctx
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot map site."
    body: dict = {
        "api_key": TAVILY_KEY,
        "url": url,
        "max_depth": max(1, min(5, int(max_depth))),
        "max_breadth": max(1, min(500, int(max_breadth))),
        "limit": max(1, int(limit)),
    }
    if instructions:
        body["instructions"] = instructions
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/map",
            json=body,
            timeout=160,
        )
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)


@function_tool
async def render_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    max_lines: int = 400,
) -> str:
    """Render a text or markdown file from the session workspace to the host terminal."""
    sid = ctx.context.session_id
    rr = ctx.context.backend.read(sid, path, 0, max(1, min(max_lines, 5000)))
    if rr.error:
        return rr.error
    text = rr.content or ""
    bar = "=" * 72
    print(f"\n{bar}\n{text}\n{bar}\n", flush=True)
    return f"Rendered {path} ({len(text)} characters) to the terminal."


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


web_agent = {
    "name": "web_agent",
    "description": (
        "Web research: give the full topic list in one call. Searches and extracts content, "
        "writes notes under /_workspace_/research/, returns file paths."
    ),
    "system_prompt": (
        "You are a web research specialist. Your tools: web_search, web_extract, web_map.\n"
        "You receive a list of research topics in a single call. Research ALL of them, then "
        "consolidate every finding into ONE markdown file at /_workspace_/research/findings.md.\n"
        "Structure: clear headings per topic, key facts in bullets, inline citations [1][2], "
        "sources section at the bottom. Do NOT paste raw tool output.\n"
        "Return exactly: 'Saved research to: /_workspace_/research/findings.md'"
    ),
    "tools": [web_search, web_extract, web_map],
    "interrupt_on": ["web_search"],
    "checkpointer": "test_demo/demo_agent.db",
}

writer = {
    "name": "writer",
    "description": (
        "Technical writer. Give it a task description and one or more research file paths. "
        "It reads the files and writes the polished document to disk, then returns that path."
    ),
    "system_prompt": (
        "You are a professional technical writer. "
        "Read the research files you are given with read_file, then write_file to "
        "/_workspace_/reports/deliverable.md with the complete document (markdown). "
        "Return exactly one line: the path /_workspace_/reports/deliverable.md"
    ),
    "checkpointer": "test_demo/demo_agent.db",
}

_chinook_db = os.path.join(_DEMO_DIR, "chinook.db")
_sql_tools = create_sql_tools(_chinook_db) if os.path.exists(_chinook_db) else []

sql_agent_runner = create_deep_agent(
    model="gpt-5-nano",
    name="sql_agent",
    description="Answers questions about the Chinook SQLite database using read-only SQL tools.",
    tools=_sql_tools,
    skills=[_SQL_SKILLS],
    system_prompt=(
        "You are the database specialist. Use sql_db_list_tables, sql_db_schema, and sql_db_query. "
        "Read the schema-exploration and query-writing skills when needed. Return clear answers."
    ),
    checkpointer="test_demo/demo_agent.db",
    backend=_DEMO_BACKEND,
)

agent = create_deep_agent(
    model="gpt-5-nano",
    name="orchestrator",
    description=(
        "General-purpose orchestrator. Handles web research, document writing, "
        "and text-to-SQL queries against the Chinook database."
    ),
    subagents=[web_agent, writer],
    tools=[*_sql_tools, render_file],
    skills=[_DEMO_DIR],
    system_prompt=(
        "You are the orchestrator. For web research + reports: delegate to web_agent_subagent then "
        "writer_subagent. For database questions: delegate to sql_agent_subagent. "
        "Pass file paths between steps; do not paste large file bodies into prompts. "
        "After the writer returns a report path, call render_file on that path so the user sees it."
    ),
    interrupt_on=["web_search"],
    checkpointer="test_demo/demo_agent.db",
    debug=True,
    backend=_DEMO_BACKEND,
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
        positional = [a for a in args if not a.startswith("-")]
        resume_sid = positional[0] if positional else os.environ.get("SESSION_ID")
        run_interactive(agent, session_id=resume_sid)
