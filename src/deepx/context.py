from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deepx.backends.protocol import BackendProtocol

if TYPE_CHECKING:
    from deepx.tools.planning import Plan


@dataclass
class AgentContext:
    session_id: str
    backend: BackendProtocol
    agent_name: str = ""
    plan: "Plan" = field(init=False)
    memory: str = ""
    skills: str = ""
    debug: bool = False
    resume: bool = False
    is_subagent: bool = False
    hitl: Any = None

    def __post_init__(self) -> None:
        from deepx.tools.planning import Plan as PlanModel

        an = self.agent_name or "agent"
        self.plan = PlanModel(session_id=self.session_id, agent_name=an)
