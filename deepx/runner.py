from dataclasses import dataclass

from agents import Runner

from deepx.context import AgentContext
from deepx.middleware.hitl import HITLHooks
from deepx.middleware.hooks import DeepRunHooks, MergedRunHooks
from deepx.session import session_factory
from deepx.skills import SkillsLoader
from deepx.storage.memory_store import MemoryStore
from deepx.storage.vfs_store import VFSStore


@dataclass
class DeepRunResult:
    output: str
    session_id: str
    vfs: dict[str, str]
    step_log: list[dict]
    token_usage: int


class DeepRunner:
    def __init__(
        self,
        agent,
        db_path,
        max_turns,
        skills_path,
        memory_path,
        hitl_tools,
        hitl_approval_fn,
    ):
        self.agent = agent
        self.db_path = db_path
        self.max_turns = max_turns
        self.skills_path = skills_path
        self.memory_path = memory_path
        self.hitl_tools = hitl_tools
        self.hitl_approval_fn = hitl_approval_fn

    async def run(self, task: str, session_id: str, resume: bool = False) -> DeepRunResult:
        ctx = AgentContext(session_id=session_id)
        if resume:
            ctx.vfs = await VFSStore(self.db_path).load(session_id)
        if self.memory_path:
            ctx.memory = MemoryStore().load(self.memory_path)
        if self.skills_path:
            skills = SkillsLoader.discover(self.skills_path)
            ctx.skills_info = SkillsLoader.format_for_prompt(skills)
        session = session_factory(session_id, self.db_path)
        if self.hitl_tools:
            hitl = HITLHooks(set(self.hitl_tools), self.hitl_approval_fn)
            hooks = MergedRunHooks(hitl)
        else:
            hooks = DeepRunHooks()
        result = await Runner.run(
            self.agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self.max_turns,
        )
        await VFSStore(self.db_path).save(ctx.vfs, session_id)
        if self.memory_path:
            MemoryStore().save(ctx.memory, self.memory_path)
        return DeepRunResult(
            output=str(result.final_output),
            session_id=session_id,
            vfs=ctx.vfs,
            step_log=ctx.step_log,
            token_usage=ctx.token_usage,
        )
