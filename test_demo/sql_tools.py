"""Minimal read-only SQLite tool factory for demo SQL subagents."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agents import RunContextWrapper, function_tool


def create_sql_tools(test_dbs_dir: Path, *, tool_prefix: str = "") -> list:
    """Read-only SQL tools.

    ``db_name`` is a ``*.db`` filename under the single allowlisted ``test_dbs_dir``.
    Typical flow: ``db_list_tables`` → ``db_schema`` for relevant tables → ``db_query`` with JOINs
    built from schema keys (avoid guessing column names).
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
    def db_list_tables(ctx: RunContextWrapper, db_name: str) -> str:
        """List all tables in the SQLite file ``db_name`` (filename under ``test_dbs``)."""
        _ = ctx
        sqlite_path = str(_resolve_db(db_name))
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return ", ".join(r[0] for r in rows) or "No tables found."

    @function_tool(name_override=f"{pre}db_schema")
    def db_schema(ctx: RunContextWrapper, db_name: str, table_names: str) -> str:
        """Show CREATE TABLE and sample rows for comma-separated ``table_names`` in ``db_name``."""
        sqlite_path = str(_resolve_db(db_name))
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

    @function_tool(name_override=f"{pre}db_query")
    def db_query(ctx: RunContextWrapper, db_name: str, query: str) -> str:
        """Run a read-only SELECT on ``db_name`` (filename under ``test_dbs``). No writes."""
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
            lines.append(" | ".join(str(row[k]) for k in keys))
        return "\n".join(lines)

    return [db_list_tables, db_schema, db_query]
