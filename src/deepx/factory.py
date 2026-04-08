from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from pathlib import Path
from typing import TypedDict

from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.result import RunResultStreaming
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
        isinstance(s, dict) and s.get("name") == "general_purpose" for s in sub_specs
    )
    if not has_gp:
        sub_specs.append(
            {
                "name": "general_purpose",
                "description": (
                    "General-purpose agent for isolated multi-step tasks. "
                    "Has access to the same filesystem and planning tools as the main agent."
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
        spaths = skills_paths if an == "general_purpose" else list(spec.get("skills", []))
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

    subagent_tools: list[Tool] = []
    for spec in sub_specs:
        an = spec["name"] if isinstance(spec, dict) else spec[0].name
        tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", an)
        subagent_tools.append(
            _make_subagent_tool(
                agent=registry[an],
                tool_name=tool_name,
                description=descriptions[an],
                db_path=db_path,
                max_turns=max_turns,
                backend=resolved_backend,
                hitl=hitl,
                debug=debug,
                memory_default=mem_content,
                skills_info=skills_by_agent.get(an, ""),
                interrupt_tools=interrupt_list,
            )
        )

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(ctx, agent, custom_prompt=system_prompt)

    main_agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=base_tools + user_tools + subagent_tools,
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
    if an == "general_purpose":
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


def _make_subagent_tool(
    *,
    agent: Agent,
    tool_name: str,
    description: str,
    db_path: str,
    max_turns: int,
    backend: BackendProtocol,
    hitl: HumanInTheLoopHooks | None,
    debug: bool,
    memory_default: str,
    skills_info: str,
    interrupt_tools: list[str],
) -> Tool:
    """Create a named FunctionTool that runs a subagent in an isolated context."""

    async def _invoke(ctx: RunContextWrapper[AgentContext], input: str) -> str:
        sub_ctx = AgentContext(
            session_id=ctx.context.session_id,
            backend=ctx.context.backend,
            agent_name=agent.name,
            memory=memory_default,
            skills_info=skills_info,
            debug=debug,
            hitl_tools=interrupt_tools,
        )
        sub_sid = f"{ctx.context.session_id}:{agent.name}:{uuid.uuid4().hex[:12]}"
        session = create_session(sub_sid, db_path)
        wrapped = apply_tool_pipeline(
            list(agent.tools),
            backend,
            agent_name=agent.name,
            debug=debug,
            hitl=hitl,
        )
        result = await Runner.run(
            dataclasses.replace(agent, tools=wrapped),
            input=input,
            context=sub_ctx,
            session=session,
            hooks=FilesystemHooks(backend),
            max_turns=max_turns,
        )
        return str(result.final_output)

    return function_tool(
        _invoke,
        name_override=tool_name,
        description_override=description,
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
        return AgentContext(
            session_id=session_id,
            backend=self._backend,
            agent_name=self._agent_name,
            memory=self._memory,
            skills_info=self._skills_info,
            debug=self._debug,
            hitl_tools=self._interrupt_tools,
            resume=resume,
        )

    def _make_hooks(self) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        return FilesystemHooks(self._backend)

    def _prepare_agent(self, ctx: AgentContext) -> Agent:
        wrapped = apply_tool_pipeline(
            list(self._agent.tools),
            self._backend,
            agent_name=self._agent.name,
            debug=self._debug,
            hitl=self._hitl,
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
