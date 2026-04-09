from __future__ import annotations

from dataclasses import dataclass, field

from deepx.backends.protocol import BackendProtocol
from deepx.models import Plan


@dataclass
class AgentContext:
    session_id: str
    backend: BackendProtocol
    agent_name: str = ""
    plan: Plan = field(init=False)
    memory: str = ""
    skills_info: str = ""
    debug: bool = False
    hitl_tools: list[str] = field(default_factory=list)
    resume: bool = False
    is_subagent: bool = False

    def __post_init__(self) -> None:
        an = self.agent_name or "agent"
        self.plan = Plan(session_id=self.session_id, agent_name=an)
