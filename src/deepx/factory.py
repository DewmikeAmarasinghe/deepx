from __future__ import annotations

import asyncio
import dataclasses
import uuid
from pathlib import Path
from typing import Any, TypedDict

from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.result import RunResultStreaming
from agents.run_context import AgentHookContext
from agents.tool import Tool

from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks, apply_tool_pipeline
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.observability import setup_observability
from deepx.models import Plan
from deepx.sessions import create_session
from deepx.system_prompt import (
    build_system_prompt,
    discover_skills,
    format_skills_for_prompt,
)
from deepx.tools import FILESYSTEM_TOOLS, PLANNING_TOOLS

_HookList = list[RunHooksBase[AgentContext, AgentType[AgentContext]]]


class SubAgentDict(TypedDict, total=False):
    name: str
    description: str
    system_prompt: str
    tools: list
    model: str
    skills: list[str]


def create_deep_agent(
    model: str = "gpt-4o-mini",
    tools: list | None = None,
    *,
    name: str = "agent",
    system_prompt: str = "",
    subagents: list[dict | tuple[Agent, str]] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: type | None = None,
    backend: BackendProtocol | None = None,
    db_path: str = ":memory:",
    interrupt_on: list[str] | None = None,
    debug: bool = False,
    max_turns: int = 1000,
) -> DeepAgentRunner:
    setup_observability()

    resolved_backend = backend or FilesystemBackend(".deepx")
    mem_content = _load_memory(memory, resolved_backend)
    skills_paths = skills or []
    skills_info_main = format_skills_for_prompt(discover_skills(skills_paths))
    user_tools = list(tools or [])
    base_tools = _build_base_tools()

    sub_specs = list(subagents or [])
    has_gp = any(
        isinstance(s, dict) and s.get("name") == "general-purpose" for s in sub_specs
    )
    if not has_gp:
        sub_specs.append(
            {
                "name": "general-purpose",
                "description": (
                    "General-purpose agent for isolated multi-step tasks. "
                    "Has access to the same tools as the main agent."
                ),
                "system_prompt": "",
                "tools": user_tools,
                "skills": skills_paths,
            }
        )

    skills_by_agent: dict[str, str] = {}
    registry: dict[str, Agent] = {}
    descriptions: dict[str, str] = {}

    for spec in sub_specs:
        if isinstance(spec, tuple):
            ag, desc = spec
            registry[ag.name] = ag
            descriptions[ag.name] = desc
            skills_by_agent[ag.name] = ""
            continue
        an = spec["name"]
        if an == "general-purpose":
            spaths = skills_paths
        else:
            spaths = list(spec.get("skills", []))
        skills_by_agent[an] = format_skills_for_prompt(discover_skills(spaths))
        descriptions[an] = spec["description"]
        registry[an] = _build_subagent_from_dict(
            spec,
            model=model,
            base_tools=base_tools,
            user_tools=user_tools,
            response_format=response_format,
        )

    hitl = HumanInTheLoopHooks(interrupt_on) if interrupt_on else None
    interrupt_list = list(interrupt_on or [])

    task_tool = _make_task_tool(
        registry=registry,
        db_path=db_path,
        max_turns=max_turns,
        backend=resolved_backend,
        hitl=hitl,
        debug=debug,
        memory_default=mem_content,
        skills_by_agent=skills_by_agent,
        interrupt_tools=interrupt_list,
        descriptions=descriptions,
    )

    for an, ag in registry.items():
        if not isinstance(ag.tools, list):
            continue
        if not any(getattr(t, "name", None) == "task" for t in ag.tools):
            registry[an] = dataclasses.replace(ag, tools=list(ag.tools) + [task_tool])

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(ctx, agent, custom_prompt=system_prompt)

    main_agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=base_tools + user_tools + [task_tool],
        output_type=response_format,
    )

    return DeepAgentRunner(
        agent=main_agent,
        backend=resolved_backend,
        db_path=db_path,
        max_turns=max_turns,
        hitl=hitl,
        skills_info=skills_info_main,
        memory=mem_content,
        debug=debug,
        agent_name=name,
        interrupt_tools=interrupt_list,
    )


def _build_subagent_from_dict(
    spec: dict,
    *,
    model: str,
    base_tools: list,
    user_tools: list,
    response_format: type | None,
) -> Agent:
    an = spec["name"]
    if an == "general-purpose":
        sub_tools = base_tools + list(spec.get("tools", user_tools))
    else:
        sub_tools = base_tools + list(spec.get("tools", []))
    sub_model = spec.get("model", model)

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(
            ctx, agent, custom_prompt=spec.get("system_prompt", "")
        )

    return Agent(
        name=an,
        instructions=instructions,
        model=sub_model,
        tools=sub_tools,
        output_type=response_format,
    )


def _make_task_tool(
    *,
    registry: dict[str, Agent],
    db_path: str,
    max_turns: int,
    backend: BackendProtocol,
    hitl: HumanInTheLoopHooks | None,
    debug: bool,
    memory_default: str,
    skills_by_agent: dict[str, str],
    interrupt_tools: list[str],
    descriptions: dict[str, str],
) -> Tool:
    available_agents = "\n".join(
        f'- "{k}": {descriptions.get(k, "")}' for k in sorted(registry)
    )
    doc = (
        "Launch an ephemeral subagent to handle complex, multi-step independent tasks "
        "with isolated context windows.\n\n"
        f"Available agent types and the tools they have access to:\n{available_agents}\n\n"
        "When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.\n\n"
        "## Usage notes:\n"
        "1. Launch multiple agents concurrently whenever possible, to maximize performance; "
        "to do that, use a single message with multiple tool uses\n"
        "2. When the agent is done, it will return a single message back to you. The result returned "
        "by the agent is not visible to the user. To show the user the result, you should send a text "
        "message back to the user with a concise summary of the result.\n"
        "3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, "
        "nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt "
        "should contain a highly detailed task description for the agent to perform autonomously and you should "
        "specify exactly what information the agent should return back to you in its final and only message to you.\n"
        "4. The agent's outputs should generally be trusted\n"
        "5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research, "
        "since it is not aware of the user's intent\n"
        "6. When only the general-purpose agent is provided, you should use it for all tasks. It is great for "
        "isolating context and token usage, and completing specific, complex tasks, as it has all the same "
        "capabilities as the main agent."
    )

    async def run_subagent_task(
        ctx: RunContextWrapper[AgentContext],
        subagent_type: str,
        description: str,
    ) -> str:
        if subagent_type not in registry:
            return f"Error: unknown subagent_type {subagent_type!r}."
        agent = registry[subagent_type]
        parent_sid = ctx.context.session_id
        sub_ctx = AgentContext(
            session_id=parent_sid,
            backend=ctx.context.backend,
            agent_name=agent.name,
            memory=memory_default,
            skills_info=skills_by_agent.get(agent.name, ""),
            debug=debug,
            hitl_tools=interrupt_tools,
        )
        sub_session_id = f"{parent_sid}:{subagent_type}:{uuid.uuid4().hex[:12]}"
        session = create_session(sub_session_id, db_path)
        hooks: _HookList = [FilesystemHooks(backend)]
        if hitl:
            hooks.append(hitl)
        combined = _CombinedHooks(hooks) if len(hooks) > 1 else hooks[0]
        wrapped_tools = apply_tool_pipeline(
            list(agent.tools),
            backend,
            agent_name=agent.name,
            debug=debug,
        )
        run_agent = dataclasses.replace(agent, tools=wrapped_tools)
        result = await Runner.run(
            run_agent,
            input=description,
            context=sub_ctx,
            session=session,
            hooks=combined,
            max_turns=max_turns,
        )
        return str(result.final_output)

    return function_tool(
        run_subagent_task,
        name_override="task",
        description_override=doc,
        use_docstring_info=False,
    )


class DeepAgentRunner:
    def __init__(
        self,
        agent: Agent,
        backend: BackendProtocol,
        db_path: str,
        max_turns: int,
        hitl: HumanInTheLoopHooks | None,
        skills_info: str,
        memory: str,
        debug: bool,
        agent_name: str,
        interrupt_tools: list[str],
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._db_path = db_path
        self._max_turns = max_turns
        self._hitl = hitl
        self._skills_info = skills_info
        self._memory = memory
        self._debug = debug
        self._agent_name = agent_name
        self._interrupt_tools = interrupt_tools

    def _make_ctx(self, session_id: str, resume: bool) -> AgentContext:
        ctx = AgentContext(
            session_id=session_id,
            backend=self._backend,
            agent_name=self._agent_name,
            memory=self._memory,
            skills_info=self._skills_info,
            debug=self._debug,
            hitl_tools=self._interrupt_tools,
        )
        if resume:
            saved = self._backend.load_plan(session_id, self._agent_name)
            if saved:
                ctx.plan = Plan.model_validate_json(saved)
        return ctx

    def _make_hooks(self) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        hooks: _HookList = [FilesystemHooks(self._backend)]
        if self._hitl:
            hooks.append(self._hitl)
        return _CombinedHooks(hooks) if len(hooks) > 1 else hooks[0]

    def _prepare_agent(self, ctx: AgentContext) -> Agent:
        wrapped = apply_tool_pipeline(
            list(self._agent.tools),
            self._backend,
            agent_name=self._agent.name,
            debug=self._debug,
        )
        return dataclasses.replace(self._agent, tools=wrapped)

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ) -> DeepRunResult:
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        if self._debug:
            self._backend.append_task_log(sid, task)
        session = create_session(sid, self._db_path)
        agent = self._prepare_agent(ctx)
        hooks = self._make_hooks()
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
    ) -> DeepRunResult:
        coro = self.run(task, session_id=session_id, resume=resume)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)  # type: ignore[return-value]
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()  # type: ignore[return-value]

    async def run_stream(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ):
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        if self._debug:
            self._backend.append_task_log(sid, task)
        session = create_session(sid, self._db_path)
        agent = self._prepare_agent(ctx)
        hooks = self._make_hooks()
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
    def __init__(self, output: str, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return (
            f"DeepRunResult(session_id={self.session_id!r}, "
            f"output={str(self.output)[:80]!r})"
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
    return [*FILESYSTEM_TOOLS, *PLANNING_TOOLS]


def _load_memory(memory: list[str] | None, backend: BackendProtocol) -> str:
    if not memory:
        return ""
    parts: list[str] = []
    for path in memory:
        p = Path(path)
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        else:
            rel = path.lstrip("/")
            raw = backend.read_store(rel)
            if raw is not None:
                parts.append(raw)
    return "\n\n".join(parts)


DeepAgent = create_deep_agent
