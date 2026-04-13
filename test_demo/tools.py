"""Shared Tavily web tools, SQL helpers, and render_file for the multi-agent demo."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx.context import AgentContext

load_dotenv()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")

DEMO_DIR = Path(__file__).resolve().parent
SKILLS_DIR = DEMO_DIR / "skills"
SQL_SKILLS_DIR = SKILLS_DIR / "sql"
PDF_SKILLS_DIR = SKILLS_DIR / "pdf"
CHINOOK_DB = DEMO_DIR / "chinook.db"
NORTHWIND_DB = DEMO_DIR / "northwind.db"
DBS_DIR = DEMO_DIR / "dbs"


def _strip_images(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj = dict(obj)
        obj.pop("images", None)
        for k, v in list(obj.items()):
            obj[k] = _strip_images(v)
        return obj
    if isinstance(obj, list):
        return [_strip_images(x) for x in obj]
    return obj


@function_tool
async def web_search(ctx: RunContextWrapper, queries: list[str]) -> str:
    """
    Search the public web using the **Tavily** API (hosted index).

    Pass **several strings in `queries`** when the topics are independent so Tavily can run
    them in parallel (one string is valid when you only need a single search). Combining
    unrelated facets into one query is cheaper but may miss coverage; splitting unrelated
    topics into multiple entries improves parallelism and recall.
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
            row = {
                "query": q,
                "answer": data.get("answer"),
                "results": data.get("results", []),
            }
            return _strip_images(row)

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
    Extract readable text from specific URLs using the **Tavily** extract API.

    Use when you already have URLs and need cleaned page text for analysis or summarization.
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
    out = _strip_images(
        {"results": data.get("results", []), "failed_results": data.get("failed_results", [])}
    )
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
    Map outbound links from a root URL using the **Tavily** map API (single-site crawl).

    Prefer conservative limits to control cost.
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
        return json.dumps(_strip_images(r.json()), indent=2)


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


def create_sql_tools(sqlite_path: str, *, tool_prefix: str = "") -> list:
    """Read-only SQL tools for a SQLite file.

    When ``tool_prefix`` is set (e.g. ``\"chinook\"``), tool names become
    ``chinook_sql_db_list_tables``, etc., so two databases can coexist in one agent.
    """
    pre = f"{tool_prefix}_" if tool_prefix else ""

    @function_tool(name_override=f"{pre}sql_db_list_tables")
    def sql_db_list_tables(ctx: RunContextWrapper) -> str:
        """List all tables in the database."""
        _ = ctx
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return ", ".join(r[0] for r in rows) or "No tables found."

    @function_tool(name_override=f"{pre}sql_db_schema")
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

    @function_tool(name_override=f"{pre}sql_db_query")
    def sql_db_query(ctx: RunContextWrapper, query: str) -> str:
        """Run a read-only SELECT. No INSERT/UPDATE/DELETE/DROP. Use LIMIT if needed."""
        _ = ctx
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
