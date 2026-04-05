from __future__ import annotations
from dataclasses import dataclass, field
from deepx.models import Plan
from deepx.backends.protocol import WorkspaceBackend


@dataclass
class AgentContext:
    session_id: str
    backend: WorkspaceBackend
    plan: Plan = field(init=False)
    memory: str = ""
    skills_info: str = ""

    def __post_init__(self) -> None:
        self.plan = Plan(session_id=self.session_id)
