from __future__ import annotations

from dataclasses import dataclass, field

from deepx.backends.protocol import WorkspaceBackend
from deepx.models import Plan


@dataclass
class AgentContext:
    """Shared state threaded through every tool call and lifecycle hook within a run."""

    session_id: str
    backend: WorkspaceBackend
    plan: Plan = field(init=False)
    memory: str = ""
    skills_info: str = ""
    approved_tools: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.plan = Plan(session_id=self.session_id)