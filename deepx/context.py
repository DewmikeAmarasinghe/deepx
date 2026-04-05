from dataclasses import dataclass, field


@dataclass
class AgentContext:
    session_id: str
    vfs: dict[str, str] = field(default_factory=dict)
    todos: list[str] = field(default_factory=list)
    visited_urls: set[str] = field(default_factory=set)
    memory: str = ""
    skills_info: str = ""
    step_log: list[dict] = field(default_factory=list)
    token_usage: int = 0
    _step_counter: int = field(default=0, repr=False)
