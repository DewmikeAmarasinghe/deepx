from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class Todo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    title: str
    status: TodoStatus = TodoStatus.pending


class Plan(BaseModel):
    session_id: str
    agent_name: str
    tasks: list[str] = Field(default_factory=list)
    todos: list[Todo] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def ensure_todo_ids(self) -> Plan:
        new: list[Todo] = []
        for i, t in enumerate(self.todos):
            nid = t.id if t.id else str(i + 1)
            new.append(t.model_copy(update={"id": nid}))
        self.todos = new
        return self

    def pending(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.pending]

    def in_progress(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.in_progress]

    def completed(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.completed]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
