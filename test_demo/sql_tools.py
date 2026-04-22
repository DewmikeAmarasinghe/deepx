"""Minimal read-only SQLite tool factory for demo SQL subagents."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agents import RunContextWrapper, function_tool


def _format_sql_value(val: object) -> str:
    """Avoid dumping binary BLOBs into model context (e.g. Northwind Picture/Photo columns)."""
    if isinstance(val, memoryview):
        val = val.tobytes()
    if isinstance(val, (bytes, bytearray)):
        return f"<BLOB {len(val)} bytes>"
    return str(val)


def _canonical_table_name(conn: sqlite3.Connection, raw: str) -> str | None:
    """Resolve user-supplied table name to a real sqlite_master name (exact or case-insensitive)."""
    key = raw.strip().strip('"').strip("`").strip()
    if not key:
        return None
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = [r[0] for r in rows]
    if key in names:
        return key
    lower_map = {n.lower(): n for n in names}
    return lower_map.get(key.lower())


def create_sql_tools(test_dbs_dir: Path, *, tool_prefix: str = "") -> list:
    """Build read-only SQLite tools for a single allowlisted directory.

    Tools:

    - **list_tables** — table names in ``db_name``.
    - **schema** — ``CREATE TABLE`` plus up to three sample rows per table (BLOBs summarized, not hex-dumped).
    - **query** — ``SELECT`` only; non-SELECT keywords rejected.

    Args:
        test_dbs_dir: Directory containing only ``*.db`` files agents may open.
        tool_prefix: Prepended to tool names (e.g. ``\"sql\"`` → ``sql_db_query``).
    """
    root = test_dbs_dir.resolve()
    pre = f"{tool_prefix}_" if tool_prefix else ""

    def _resolve_db(db_name: str) -> Path:
        name = (db_name or "").strip()
        if not name or ".." in name or "/" in name or "\\" in name:
            raise ValueError(
                "Invalid db_name: use a plain filename (e.g. chinook.db or northwind.db)."
            )
        if not name.endswith(".db"):
            name = f"{name}.db"
        p = (root / name).resolve()
        if p.parent != root:
            raise ValueError(
                "db_name must resolve to a file directly under the allowlisted test_dbs dir."
            )
        if not p.is_file():
            raise ValueError(f"Database not found under test_dbs: {name}")
        return p

    @function_tool(name_override=f"{pre}db_list_tables")
    async def db_list_tables(ctx: RunContextWrapper, db_name: str) -> str:
        """Discover all available user-defined tables in the SQLite database.

        Use this tool FIRST when working with a database unless the table names are
        already known from prior tool outputs.

        This helps you:
        - identify which tables actually exist
        - avoid hallucinating table names
        - decide which tables to inspect next with db_schema

        Typical workflow:
        1. call db_list_tables
        2. inspect relevant tables with db_schema
        3. write SQL using db_query

        Input:
        - db_name: database filename (for example: "chinook.db")

        Output:
        - comma-separated list of valid table names
        """
        _ = ctx
        sqlite_path = str(_resolve_db(db_name))
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return ", ".join(r[0] for r in rows) or "No tables found."

    @function_tool(name_override=f"{pre}db_schema")
    async def db_schema(ctx: RunContextWrapper, db_name: str, table_names: str) -> str:
        """Inspect the schema and sample data for one or more tables.

        Use this tool AFTER db_list_tables and BEFORE writing SQL queries.

        This tool helps you understand:
        - exact column names
        - column order
        - data types
        - table relationships
        - realistic sample values

        This is essential before using db_query because SQL should reference
        real columns exactly as defined in the schema.

        Input:
        - db_name: database filename
        - table_names: comma-separated table names

        Example:
        table_names="customers, invoices"

        Output includes:
        - CREATE TABLE statement
        - column definitions
        - up to 3 sample rows per table
        - binary data summarized as <BLOB n bytes>

        Use sample rows to infer join keys and filtering values.
        """
        _ = ctx
        sqlite_path = str(_resolve_db(db_name))
        raw_names = [n.strip() for n in table_names.split(",") if n.strip()]
        parts: list[str] = []
        with sqlite3.connect(sqlite_path) as conn:
            for fragment in raw_names:
                canonical = _canonical_table_name(conn, fragment)
                if not canonical:
                    parts.append(f"Table '{fragment}' not found.")
                    continue
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (canonical,),
                ).fetchone()
                if not row:
                    parts.append(f"Table '{fragment}' not found.")
                    continue
                schema = row[0]
                qident = canonical.replace('"', '""')
                sample_rows = conn.execute(
                    f'SELECT * FROM "{qident}" LIMIT 3'
                ).fetchall()
                cols = [
                    d[0]
                    for d in conn.execute(
                        f'SELECT * FROM "{qident}" LIMIT 0'
                    ).description
                ]
                sample = "\n".join(
                    ", ".join(f"{c}={_format_sql_value(v)}" for c, v in zip(cols, r))
                    for r in sample_rows
                )
                parts.append(f"-- {canonical}\n{schema}\n\n-- Sample rows:\n{sample}")
        return "\n\n".join(parts)

    @function_tool(name_override=f"{pre}db_query")
    async def db_query(ctx: RunContextWrapper, db_name: str, query: str) -> str:
        """Execute a read-only SQL SELECT query against the SQLite database.

        IMPORTANT:
        Use this tool ONLY after inspecting the schema with db_schema unless
        the required tables and columns are already confirmed.

        Allowed:
        - SELECT
        - JOIN
        - WHERE
        - GROUP BY
        - ORDER BY
        - LIMIT
        - aggregate functions (COUNT, SUM, AVG, etc.)

        Not allowed:
        - INSERT
        - UPDATE
        - DELETE
        - DROP
        - CREATE
        - ALTER

        Best practices:
        - always use exact table and column names from db_schema
        - prefer LIMIT for exploratory queries
        - use JOIN when combining related tables
        - verify column names before querying
        - if query fails, inspect schema again

        Input:
        - db_name: database filename
        - query: SQL SELECT statement

        Output:
        - pipe-delimited result table
        - "Query returned no rows" if empty
        """
        _ = ctx
        sqlite_path = str(_resolve_db(db_name))
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
            lines.append(" | ".join(_format_sql_value(row[k]) for k in keys))
        return "\n".join(lines)

    return [db_list_tables, db_schema, db_query]
