from __future__ import annotations

import asyncio
import dataclasses
import os
import uuid
from pathlib import Path
from typing import Any, TypedDict

from agents import Agent, RunContextWrapper, Runner
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.result import RunResultStreaming
from agents.run_context import AgentHookContext
from agents.tool import Tool

from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory_backend import InMemoryBackend
from deepx.backends.protocol import WorkspaceBackend
from deepx.context import AgentContext
from deepx.instructions import BASE_PROMPT, build_instructions
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.sessions import create_session
from deepx.middleware.skills import discover_skills, format_skills_for_prompt
from deepx.middleware.workspace import WorkspaceHooks, wrap_tools_for_logging
from deepx.models import Plan

_HookList = list[RunHooksBase[AgentContext, AgentType[AgentContext]]]


class SubAgentDict(TypedDict, total=False):
    """Dict-based subagent specification, mirrors the langchain/deepagents API."""

    name: str
    description: str
    system_prompt: str
    tools: list
    model: str
    skills: list[str]


def create_deep_agent(
    *,
    model: str = "gpt-4o-mini",
    tools: list | None = None,
    subagents: list[SubAgentDict | tuple[Agent, str]] | None = None,
    system_prompt: str = "",
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    workspace_path: str = ".deepx",
    db_path: str = ":memory:",
    max_turns: int = 200,
    require_approval: list[str] | None = None,
    log_tools: bool = False,
) -> "DeepAgent":
    """Create a DeepAgent — an autonomous agent with planning, workspace, and memory.

    Args:
        model: OpenAI model identifier for the orchestrator.
        tools: Custom tools available to the orchestrator and the general-purpose subagent.
        subagents: List of subagent definitions.  Each item is either a
            ``SubAgentDict`` (dict with ``name``, ``description``, ``system_prompt``,
            ``tools``, optional ``model`` and ``skills``) or a
            ``(Agent, description)`` tuple for pre-built agents.
        system_prompt: Additional instructions prepended to the orchestrator's prompt.
        skills: Paths to skill directories (each must contain ``SKILL.md`` files).
            The orchestrator and general-purpose subagent share these skills.
        memory: Paths to ``AGENTS.md`` files loaded as shared memory.
        workspace_path: Root directory for all workspace files.  Defaults to ``.deepx``.
            Set ``DEEPX_WORKSPACE`` env var to override globally.
        db_path: SQLite database path for session persistence.  Defaults to
            ``:memory:`` (ephemeral).  Use ``"agent.db"`` for persistent sessions.
        max_turns: Maximum number of LLM turns before the run is halted.
        require_approval: Tool names that require human approval before each call.
            Once approved, the tool is remembered for the rest of the session.
        log_tools: Write a JSON log file per tool call (input + output) to the workspace.
    """
    from deepx.observability import setup_observability
    setup_observability()

    workspace_root = workspace_path or os.getenv("DEEPX_WORKSPACE", ".deepx")
    backend: WorkspaceBackend = (
        FilesystemBackend(workspace_root) if workspace_root else InMemoryBackend()
    )

    mem_content = _load_memory(memory, backend)
    skills_info = format_skills_for_prompt(discover_skills(skills or []))

    user_tools: list = tools or []
    base_tools = _build_base_tools()
    all_tools = base_tools + user_tools

    subagent_tools = _build_subagent_tools(subagents or [], model, all_tools)
    all_tools_with_subs = all_tools + subagent_tools

    spawn_tool = _make_spawn_tool(model, all_tools_with_subs)
    final_tools = all_tools_with_subs + [spawn_tool]

    hitl = HumanInTheLoopHooks(require_approval) if require_approval else None

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_instructions(ctx, agent, custom_prompt=system_prompt)

    orchestrator = Agent(
        name="orchestrator",
        instructions=instructions,
        model=model,
        tools=final_tools,
    )

    return DeepAgent(
        agent=orchestrator,
        backend=backend,
        db_path=db_path,
        max_turns=max_turns,
        hitl=hitl,
        skills_info=skills_info,
        memory=mem_content,
        log_tools=log_tools,
    )


class DeepAgent:
    """The runnable agent returned by ``create_deep_agent``."""

    def __init__(
        self,
        agent: Agent,
        backend: WorkspaceBackend,
        db_path: str,
        max_turns: int,
        hitl: HumanInTheLoopHooks | None,
        skills_info: str,
        memory: str,
        log_tools: bool,
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._db_path = db_path
        self._max_turns = max_turns
        self._hitl = hitl
        self._skills_info = skills_info
        self._memory = memory
        self._log_tools = log_tools

    def _make_ctx(self, session_id: str, resume: bool) -> AgentContext:
        ctx = AgentContext(session_id=session_id, backend=self._backend)
        ctx.memory = self._memory
        ctx.skills_info = self._skills_info
        if resume:
            saved = self._backend.load_plan(session_id)
            if saved:
                ctx.plan = Plan.model_validate_json(saved)
        return ctx

    def _make_hooks(
        self, should_log: bool
    ) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        hooks: _HookList = [WorkspaceHooks(self._backend, should_log)]
        if self._hitl:
            hooks.append(self._hitl)
        return _CombinedHooks(hooks) if len(hooks) > 1 else hooks[0]

    def _make_agent(self, should_log: bool, session_id: str) -> Agent:
        if not should_log:
            return self._agent
        logged = wrap_tools_for_logging(
            self._agent.tools, self._backend, session_id, self._agent.name
        )
        return dataclasses.replace(self._agent, tools=logged)

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
        log_tools: bool | None = None,
    ) -> "DeepRunResult":
        """Run the agent on *task* and return when a final output is produced.

        Args:
            task: The user's request.
            session_id: Unique thread identifier.  Auto-generated if omitted.
            resume: If ``True``, reload the persisted plan from a previous run.
            log_tools: Override the instance-level ``log_tools`` setting for this run.
        """
        sid = session_id or uuid.uuid4().hex
        should_log = log_tools if log_tools is not None else self._log_tools
        ctx = self._make_ctx(sid, resume)
        session = create_session(sid, self._db_path)
        agent = self._make_agent(should_log, sid)
        hooks = self._make_hooks(should_log)

        result = await Runner.run(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self._max_turns,
        )
        return DeepRunResult(output=result.final_output, session_id=sid, plan=ctx.plan)

    def run_sync(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
        log_tools: bool | None = None,
    ) -> "DeepRunResult":
        """Synchronous wrapper around :meth:`run`.  Safe to call from scripts and CLIs."""
        coro = self.run(task, session_id=session_id, resume=resume, log_tools=log_tools)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()

    async def run_stream(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
        log_tools: bool | None = None,
    ):
        """Async generator that yields ``StreamEvent`` objects as they arrive.

        Usage::

            async for event in agent.run_stream(task, session_id="x"):
                if event.type == "raw_response_event":
                    print(event.data, end="", flush=True)
        """
        sid = session_id or uuid.uuid4().hex
        should_log = log_tools if log_tools is not None else self._log_tools
        ctx = self._make_ctx(sid, resume)
        session = create_session(sid, self._db_path)
        agent = self._make_agent(should_log, sid)
        hooks = self._make_hooks(should_log)

        stream: RunResultStreaming = Runner.run_streamed(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self._max_turns,
        )
        async for event in stream.stream_events():
            yield event


class DeepRunResult:
    """Returned by every ``run`` method."""

    def __init__(self, output: str, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return (
            f"DeepRunResult(session_id={self.session_id!r}, "
            f"output={self.output[:80]!r})"
        )


class _CombinedHooks(RunHooksBase[AgentContext, AgentType[AgentContext]]):
    def __init__(self, hooks: _HookList) -> None:
        self._hooks = hooks

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: AgentType[AgentContext],
    ) -> None:
        for h in self._hooks:
            await h.on_agent_start(context, agent)

    async def on_agent_end(
        self,
        context: AgentHookContext[AgentContext],
        agent: AgentType[AgentContext],
        output: Any,
    ) -> None:
        for h in self._hooks:
            await h.on_agent_end(context, agent, output)

    async def on_tool_start(
        self,
        context: RunContextWrapper[AgentContext],
        agent: AgentType[AgentContext],
        tool: Tool,
    ) -> None:
        for h in self._hooks:
            await h.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: AgentType[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        for h in self._hooks:
            await h.on_tool_end(context, agent, tool, result)


def _build_base_tools() -> list:
    from deepx.tools import MEMORY_TOOLS, PLANNING_TOOLS, WORKSPACE_TOOLS
    return [*WORKSPACE_TOOLS, *PLANNING_TOOLS, *MEMORY_TOOLS]


def _load_memory(memory: list[str] | None, backend: WorkspaceBackend) -> str:
    if not memory:
        return ""
    parts: list[str] = []
    for path in memory:
        p = Path(path)
        if p.exists():
            parts.append(p.read_text())
    return "\n\n".join(parts)


def _make_spawn_tool(model: str, tools: list) -> Any:
    general_purpose = Agent(
        name="general_purpose",
        instructions=BASE_PROMPT,
        model=model,
        tools=tools,
    )
    return general_purpose.as_tool(
        tool_name="spawn_task",
        tool_description=(
            "Delegate a self-contained task to an isolated general-purpose subagent "
            "that has access to all the same tools.  Use when work would produce large "
            "intermediate output or can run independently to keep your context clean.  "
            "Pass file paths in the instructions, never raw content.  "
            "Returns only the subagent's final output."
        ),
    )


def _build_subagent_tools(
    subagents: list[SubAgentDict | tuple[Agent, str]],
    default_model: str,
    parent_tools: list,
) -> list:
    result: list = []
    for spec in subagents:
        if isinstance(spec, tuple):
            agent_obj, description = spec
            result.append(
                agent_obj.as_tool(
                    tool_name=agent_obj.name,
                    tool_description=description,
                )
            )
        else:
            name = spec["name"]
            description = spec["description"]
            system_prompt = spec.get("system_prompt", BASE_PROMPT)
            sub_tools = list(spec.get("tools", parent_tools))
            sub_model = spec.get("model", default_model)
            sub_skills = spec.get("skills", [])

            skills_info = format_skills_for_prompt(discover_skills(sub_skills))

            def _make_instructions(sp: str, si: str) -> str:
                if si:
                    return f"{sp}\n\n## Available Skills\n{si}"
                return sp

            sub_agent = Agent(
                name=name,
                instructions=_make_instructions(system_prompt, skills_info),
                model=sub_model,
                tools=sub_tools,
            )
            result.append(
                sub_agent.as_tool(
                    tool_name=name,
                    tool_description=description,
                )
            )
    return result