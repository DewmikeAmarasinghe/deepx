import json
from datetime import UTC, datetime

import aiosqlite


class VFSStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def _ensure_tables(self, db):
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS vfs_files (
                session_id TEXT,
                path TEXT,
                content TEXT,
                modified_at TEXT,
                PRIMARY KEY (session_id, path)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS step_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                step INTEGER,
                data_json TEXT
            )
            """
        )
        await db.commit()

    async def save(self, vfs: dict, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_tables(db)
            now = datetime.now(UTC).isoformat()
            for path, content in vfs.items():
                await db.execute(
                    "INSERT OR REPLACE INTO vfs_files VALUES (?,?,?,?)",
                    (session_id, path, content, now),
                )
            await db.commit()

    async def load(self, session_id: str) -> dict[str, str]:
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute(
                "SELECT path, content FROM vfs_files WHERE session_id=?",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def save_step(self, session_id: str, step: int, data: dict):
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_tables(db)
            await db.execute(
                "INSERT INTO step_log (session_id, step, data_json) VALUES (?,?,?)",
                (session_id, step, json.dumps(data)),
            )
            await db.commit()
