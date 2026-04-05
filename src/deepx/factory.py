from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path

from agents import Agent, RunContextWrapper, Runner
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext
from agents.tool import Tool

from deepx.agents_loader import AgentsLoader
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory_backend import InMemoryBackend
from deepx.backends.protocol import WorkspaceBackend
from deepx.context import AgentContext
from deepx.instructions import BASE_PROMPT, build_instructions
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.workspace import WorkspaceHooks, wrap_tools_for_logging
from deepx.models import Plan
from deepx.sessions import create_session
from deepx.skills import SkillsLoader
from deepx.tools import MEMORY_TOOLS, PLANNING_TOOLS, WORKSPACE_TOOLS


def create_deep_agent(
    *,
    model: str = "gpt-4o-mini",
    tools: list | None = None,
    subagents: list[tuple[Agent, str]] | None = None,
    agents_path: str | None = None,
    system_prompt: str = "",
    skills_path: str | None = None,
    memory_path: str | None = None,
    workspace_path: str | None = None,
    db_path: str = "agent.db",
    max_turns: int = 200,
    require_approval: list[str] | None = None,
    log_tools: bool = False,
) -> "DeepAgent":
    from deepx.observability import setup_observability
    setup_observability()

    workspace_root = workspace_path or os.getenv("DEEPX_WORKSPACE", ".deepx")
    backend: WorkspaceBackend = (
        FilesystemBackend(workspace_root) if workspace_root else InMemoryBackend()
    )

    skills_info = ""
    if skills_path:
        skills = SkillsLoader.discover(skills_path)
        skills_info = SkillsLoader.format_for_prompt(skills)

    mem_content = ""
    if memory_path:
        p = Path(memory_path)
        mem_content = p.read_text() if p.exists() else ""

    base_tools = [*WORKSPACE_TOOLS, *PLANNING_TOOLS, *MEMORY_TOOLS]
    user_tools = tools or []
    all_tools = base_tools + user_tools

    subagent_tools = [
        sub.as_tool(tool_name=sub.name, tool_description=desc)
        for sub, desc in (subagents or [])
    ]

    md_subagent_tools = []
    if agents_path:
        for defn in AgentsLoader.discover(agents_path):
            sub = Agent(
                name=defn["name"],
                instructions=defn["instructions"],
                model=model,
                tools=all_tools,
            )
            md_subagent_tools.append(
                sub.as_tool(
                    tool_name=defn["name"],
                    tool_description=defn["description"],
                )
            )

    all_tools_with_subs = all_tools + subagent_tools + md_subagent_tools

    general_purpose = Agent(
        name="general_purpose",
        instructions=BASE_PROMPT,
        model=model,
        tools=all_tools_with_subs,
    )
    spawn_tool = general_purpose.as_tool(
        tool_name="spawn_task",
        tool_description=(
            "Delegate a self-contained task to an isolated general-purpose subagent "
            "that has access to all the same tools. Use when work produces large output "
            "or can run independently. Pass file paths in instructions, not raw content. "
            "Returns only the subagent's final output."
        ),
    )

    final_tools = all_tools_with_subs + [spawn_tool]
    hitl_hooks = HumanInTheLoopHooks(require_approval) if require_approval else None

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_instructions(ctx, agent, custom_prompt=system_prompt)

    agent = Agent(
        name="orchestrator",
        instructions=instructions,
        model=model,
        tools=final_tools,
    )

    return DeepAgent(
        agent=agent,
        backend=backend,
        db_path=db_path,
        max_turns=max_turns,
        hitl_hooks=hitl_hooks,
        skills_info=skills_info,
        memory=mem_content,
        log_tools=log_tools,
    )


class DeepAgent:
    def __init__(
        self,
        agent: Agent,
        backend: WorkspaceBackend,
        db_path: str,
        max_turns: int,
        hitl_hooks: HumanInTheLoopHooks | None,
        skills_info: str,
        memory: str,
        log_tools: bool,
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._db_path = db_path
        self._max_turns = max_turns
        self._hitl_hooks = hitl_hooks
        self._skills_info = skills_info
        self._memory = memory
        self._log_tools = log_tools

    def _build_ctx(self, session_id: str, task: str, resume: bool) -> AgentContext:
        ctx = AgentContext(session_id=session_id, backend=self._backend)
        ctx.memory = self._memory
        ctx.skills_info = self._skills_info
        self._backend.write(session_id, "../task.md", task)
        if resume:
            saved = self._backend.load_plan(session_id)
            if saved:
                ctx.plan = Plan.model_validate_json(saved)
        return ctx

    def _build_hooks(self, should_log: bool) -> RunHooksBase:
        hooks_list: list[RunHooksBase] = [WorkspaceHooks(self._backend, should_log)]
        if self._hitl_hooks:
            hooks_list.append(self._hitl_hooks)
        return _CombinedHooks(hooks_list) if len(hooks_list) > 1 else hooks_list[0]

    def _maybe_wrap_agent(self, should_log: bool, session_id: str) -> Agent:
        if not should_log:
            return self._agent
        logged_tools = wrap_tools_for_logging(self._agent.tools, self._backend, session_id)
        return dataclasses.replace(self._agent, tools=logged_tools)

    async def run(
        self,
        task: str,
        *,
        session_id: str,
        resume: bool = False,
        log_tools: bool | None = None,
    ) -> "DeepRunResult":
        should_log = log_tools if log_tools is not None else self._log_tools
        ctx = self._build_ctx(session_id, task, resume)
        session = create_session(session_id, self._db_path)
        agent = self._maybe_wrap_agent(should_log, session_id)
        hooks = self._build_hooks(should_log)

        result = await Runner.run(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self._max_turns,
        )
        return DeepRunResult(output=result.final_output, session_id=session_id, plan=ctx.plan)

    def run_sync(
        self,
        task: str,
        *,
        session_id: str,
        resume: bool = False,
        log_tools: bool | None = None,
    ) -> "DeepRunResult":
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.run(task, session_id=session_id, resume=resume, log_tools=log_tools)
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    async def run_stream(
        self,
        task: str,
        *,
        session_id: str,
        resume: bool = False,
        log_tools: bool | None = None,
    ):
        """Async generator yielding raw SDK stream events.

        Usage:
            async for event in agent.run_stream(task, session_id="x"):
                if event.type == "raw_response_event":
                    delta = event.data.response.output[0].content[0].text
                    print(delta, end="", flush=True)
        """
        should_log = log_tools if log_tools is not None else self._log_tools
        ctx = self._build_ctx(session_id, task, resume)
        session = create_session(session_id, self._db_path)
        agent = self._maybe_wrap_agent(should_log, session_id)
        hooks = self._build_hooks(should_log)

        async with Runner.run_streamed(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self._max_turns,
        ) as stream:
            async for event in stream.stream_events():
                yield event


class DeepRunResult:
    def __init__(self, output: str, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return f"DeepRunResult(session_id={self.session_id!r}, output={self.output[:100]!r})"


class _CombinedHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(self, hooks: list[RunHooksBase]) -> None:
        self._hooks = hooks

    async def on_agent_start(
        self, context: AgentHookContext[AgentContext], agent: Agent[AgentContext]
    ) -> None:
        for h in self._hooks:
            await h.on_agent_start(context, agent)

    async def on_agent_end(
        self, context: AgentHookContext[AgentContext], agent: Agent[AgentContext], output
    ) -> None:
        for h in self._hooks:
            await h.on_agent_end(context, agent, output)

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
    ) -> None:
        for h in self._hooks:
            await h.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        for h in self._hooks:
            await h.on_tool_end(context, agent, tool, result)