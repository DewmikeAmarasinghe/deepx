from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path

from agents import Agent, RunContextWrapper, RunHooks, Runner

from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory_backend import InMemoryBackend
from deepx.backends.protocol import WorkspaceBackend
from deepx.context import AgentContext
from deepx.instructions import build_instructions
from deepx.middleware.workspace import WorkspaceHooks
from deepx.models import Plan
from deepx.observability import setup_observability
from deepx.sessions.factory import create_session
from deepx.skills import SkillsLoader
from deepx.tools import MEMORY_TOOLS, PLANNING_TOOLS, WORKSPACE_TOOLS

setup_observability()


def create_deep_agent(
    *,
    model: str = "gpt-4o",
    tools: list | None = None,
    subagents: list[tuple[Agent, str]] | None = None,
    system_prompt: str = "",
    skills_path: str | None = None,
    memory_path: str | None = None,
    workspace_path: str | None = None,
    db_path: str | None = None,
    max_turns: int = 200,
    hitl_hooks: RunHooks | None = None,
) -> "DeepAgent":
    workspace_root = workspace_path or os.getenv("DEEPX_WORKSPACE", ".deepx")
    backend: WorkspaceBackend = (
        FilesystemBackend(workspace_root) if workspace_root else InMemoryBackend()
    )

    if skills_path:
        skills = SkillsLoader.discover(skills_path)
        skills_info = SkillsLoader.format_for_prompt(skills)
    else:
        skills_info = ""

    if memory_path:
        mem_content = (
            Path(memory_path).read_text() if Path(memory_path).exists() else ""
        )
    else:
        mem_content = ""

    subagent_tools = []
    for sub_agent, description in subagents or []:
        subagent_tools.append(_make_subagent_tool(sub_agent, description))

    base_tools = [*WORKSPACE_TOOLS, *PLANNING_TOOLS, *MEMORY_TOOLS]
    all_tools = base_tools + subagent_tools + (tools or [])

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_instructions(ctx, agent, custom_prompt=system_prompt)

    agent = Agent(
        name="orchestrator",
        instructions=instructions,
        model=model,
        tools=all_tools,
    )

    return DeepAgent(
        agent=agent,
        backend=backend,
        db_path=db_path,
        max_turns=max_turns,
        hitl_hooks=hitl_hooks,
        skills_info=skills_info,
        memory=mem_content,
    )


class DeepAgent:
    def __init__(
        self,
        agent: Agent,
        backend: WorkspaceBackend,
        db_path: str | None,
        max_turns: int,
        hitl_hooks: RunHooks | None,
        skills_info: str,
        memory: str,
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._db_path = db_path
        self._max_turns = max_turns
        self._hitl_hooks = hitl_hooks
        self._skills_info = skills_info
        self._memory = memory

    async def run(
        self,
        task: str,
        *,
        session_id: str,
        resume: bool = False,
    ) -> "DeepRunResult":
        ctx = AgentContext(
            session_id=session_id,
            backend=self._backend,
        )
        ctx.memory = self._memory
        ctx.skills_info = self._skills_info

        if resume:
            saved_plan = self._backend.load_plan(session_id)
            if saved_plan:
                ctx.plan = Plan.model_validate_json(saved_plan)

        session = create_session(session_id, self._db_path)

        hooks_list: list[RunHooks] = [WorkspaceHooks(self._backend)]
        if self._hitl_hooks:
            hooks_list.append(self._hitl_hooks)

        combined_hooks = (
            _CombinedHooks(hooks_list) if len(hooks_list) > 1 else hooks_list[0]
        )

        result = await Runner.run(
            self._agent,
            input=task,
            context=ctx,
            session=session,
            hooks=combined_hooks,
            max_turns=self._max_turns,
        )

        return DeepRunResult(
            output=result.final_output,
            session_id=session_id,
            plan=ctx.plan,
        )

    def run_sync(
        self, task: str, *, session_id: str, resume: bool = False
    ) -> "DeepRunResult":
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.run(task, session_id=session_id, resume=resume)
                )
                return future.result()
        else:
            return asyncio.run(self.run(task, session_id=session_id, resume=resume))


class DeepRunResult:
    def __init__(self, output: str, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return f"DeepRunResult(session_id={self.session_id!r}, output={self.output[:100]!r})"


def _make_subagent_tool(sub_agent: Agent, description: str):
    from agents import function_tool

    @function_tool
    async def run_subagent(ctx: RunContextWrapper, task_description: str) -> str:
        result = await Runner.run(
            sub_agent, input=task_description, context=ctx.context
        )
        return result.final_output

    return dataclasses.replace(run_subagent, name=sub_agent.name, description=description)


class _CombinedHooks(RunHooks):
    def __init__(self, hooks: list[RunHooks]) -> None:
        self._hooks = hooks

    async def on_agent_start(self, ctx, agent) -> None:
        for h in self._hooks:
            await h.on_agent_start(ctx, agent)

    async def on_agent_end(self, ctx, agent, output) -> None:
        for h in self._hooks:
            await h.on_agent_end(ctx, agent, output)

    async def on_tool_start(self, ctx, agent, tool) -> None:
        for h in self._hooks:
            await h.on_tool_start(ctx, agent, tool)

    async def on_tool_end(self, ctx, agent, tool, result) -> None:
        for h in self._hooks:
            await h.on_tool_end(ctx, agent, tool, result)