from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone


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


class ToolLog(BaseModel):
    call_id: str
    tool_name: str
    agent_name: str
    session_id: str
    timestamp: str
    input_preview: str
    output_preview: str
    output_chars: int
    saved_to: str | None = None
