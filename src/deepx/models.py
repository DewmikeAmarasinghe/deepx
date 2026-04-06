from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class Todo(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    status: TodoStatus = TodoStatus.pending
    notes: str = ""


class Plan(BaseModel):
    """The agent's execution plan, persisted to disk after every mutation."""

    session_id: str
    todos: list[Todo] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def pending(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.pending]

    def in_progress(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.in_progress]

    def completed(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.completed]

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)