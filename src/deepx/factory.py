from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.result import RunResultStreaming
from agents.tool import Tool

from deepx.backends.filesystem import FilesystemBackend, resolve_host_root
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks, apply_tool_pipeline
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.observability import setup_observability
from deepx.tools.planning import Plan
from deepx.sessions import create_session
from deepx.system_prompt import (
    build_system_prompt,
    format_skills_for_prompt,
    skills_catalog_for_host,
)
from deepx.tools import FILESYSTEM_TOOLS, MEMORY_TOOLS, PLANNING_TOOLS

SubAgentDict = dict

HitlApprovalFn = Callable[[str, str, str], bool | Awaitable[bool]]


def _collect_skill_roots(main: list[str] | None) -> list[str]:
    """Skill roots for the *main* agent only (no merging nested subagent skill trees)."""
    out: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        rp = str(Path(raw).expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    for raw in main or []:
        add(str(raw))
    dx = Path.cwd() / ".deepx" / "skills"
    if dx.is_dir():
        add(str(dx.resolve()))
    u = Path.home() / ".deepx" / "skills"
    if u.is_dir():
        add(str(u.resolve()))
    return out


def _make_hitl(
    interrupt_tools: list[str],
    *,
    approval_fn: HitlApprovalFn | None = None,
) -> HumanInTheLoopHooks | None:
    if not interrupt_tools:
        return None
    return HumanInTheLoopHooks(list(interrupt_tools), approval_fn=approval_fn)


def _skills_prompt_for_backend(
    backend: BackendProtocol, skill_roots: list[str]
) -> str:
    host = resolve_host_root(backend)
    if host is None:
        return ""
    meta = skills_catalog_for_host(host, skill_roots)
    return format_skills_for_prompt(meta)


def create_deep_agent(
    model: str = "gpt-5-nano",
    tools: list | None = None,
    *,
    name: str = "agent",
    description: str = "",
    system_prompt: str = "",
    subagents: list[dict | "DeepAgentRunner"] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: type | None = None,
    backend: BackendProtocol | None = None,
    checkpointer: str = ":memory:",
    interrupt_on: list[str] | None = None,
    debug: bool = False,
    max_turns: int = 1000,
    hitl_approval_fn: HitlApprovalFn | None = None,
) -> "DeepAgentRunner":
    setup_observability()

    checkpointer = checkpointer.strip() or ":memory:"
    sub_specs = list(subagents or [])
    has_gp = any(
        (isinstance(s, dict) and s.get("name") == "general_purpose")
        or (isinstance(s, DeepAgentRunner) and s._agent_name == "general_purpose")
        for s in sub_specs
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
                "tools": list(tools or []),
                "skills": [],
            }
        )

    skill_roots = _collect_skill_roots(skills)
    if backend is None:
        resolved_backend = FilesystemBackend(Path.home())
    else:
        resolved_backend = backend
    mem_content = _load_memory(memory, resolved_backend)
    user_tools = list(tools or [])
    base_tools = [*FILESYSTEM_TOOLS, *MEMORY_TOOLS, *PLANNING_TOOLS]
    interrupt_list = list(interrupt_on or [])

    subagent_tools: list[Tool] = []
    for spec in sub_specs:
        runner = _resolve_subagent_spec(
            spec,
            parent_model=model,
            parent_backend=resolved_backend,
            checkpointer=checkpointer,
            debug=debug,
            mem_content=mem_content,
            skills_paths=list(skills or []),
            user_tools=user_tools,
            parent_interrupt=interrupt_list,
            max_turns=max_turns,
            parent_response_format=response_format,
            hitl_approval_fn=hitl_approval_fn,
        )
        an = runner._agent_name
        tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", an)
        subagent_tools.append(
            _make_subagent_tool(
                runner=runner,
                tool_name=tool_name,
                checkpointer=runner._checkpointer,
                max_turns=max_turns,
                backend=resolved_backend,
                debug=debug,
            )
        )

    _main_checkpointer = checkpointer

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(
            ctx, agent, custom_prompt=system_prompt, checkpointer=_main_checkpointer
        )

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
        checkpointer=checkpointer,
        max_turns=max_turns,
        skill_roots=skill_roots,
        memory=mem_content,
        debug=debug,
        agent_name=name,
        description=description,
        interrupt_tools=interrupt_list,
        hitl_approval_fn=hitl_approval_fn,
    )


def _resolve_subagent_spec(
    spec: dict | "DeepAgentRunner",
    *,
    parent_model: str,
    parent_backend: BackendProtocol,
    checkpointer: str,
    debug: bool,
    mem_content: str,
    skills_paths: list[str],
    user_tools: list,
    parent_interrupt: list[str],
    max_turns: int,
    parent_response_format: type | None,
    hitl_approval_fn: HitlApprovalFn | None,
) -> "DeepAgentRunner":
    if isinstance(spec, DeepAgentRunner):
        return spec

    an = spec.get("name", "")
    is_gp = an == "general_purpose"

    spec_skills = skills_paths if is_gp else list(spec.get("skills", []))
    spec_user_tools = list(spec.get("tools", user_tools if is_gp else []))
    spec_interrupt = list(spec.get("interrupt_on") or (parent_interrupt if is_gp else []))
    spec_memory_paths: list[str] | None = spec.get("memory")
    spec_response_format = spec.get("response_format", parent_response_format)
    spec_model = spec.get("model", parent_model)

    spec_mem = (
        _load_memory(spec_memory_paths, parent_backend)
        if spec_memory_paths
        else (mem_content if is_gp else "")
    )

    sub_roots = [str(Path(p).expanduser().resolve()) for p in spec_skills]

    base_tools = [*FILESYSTEM_TOOLS, *MEMORY_TOOLS, *PLANNING_TOOLS]

    nested_subagent_tools: list[Tool] = []
    for nested in spec.get("subagents") or []:
        nested_runner = _resolve_subagent_spec(
            nested,
            parent_model=spec_model,
            parent_backend=parent_backend,
            checkpointer=checkpointer,
            debug=debug,
            mem_content=spec_mem,
            skills_paths=spec_skills,
            user_tools=spec_user_tools,
            parent_interrupt=spec_interrupt,
            max_turns=max_turns,
            parent_response_format=spec_response_format,
            hitl_approval_fn=hitl_approval_fn,
        )
        nested_tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", nested_runner._agent_name)
        nested_subagent_tools.append(
            _make_subagent_tool(
                runner=nested_runner,
                tool_name=nested_tool_name,
                checkpointer=nested_runner._checkpointer,
                max_turns=max_turns,
                backend=parent_backend,
                debug=debug,
            )
        )

    custom_prompt = spec.get("system_prompt", "")
    _spec_checkpointer = str(spec.get("checkpointer", checkpointer))

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(
            ctx, agent, custom_prompt=custom_prompt, checkpointer=_spec_checkpointer
        )

    agent = Agent(
        name=an,
        instructions=instructions,
        model=spec_model,
        tools=base_tools + spec_user_tools + nested_subagent_tools,
        output_type=spec_response_format,
    )

    return DeepAgentRunner(
        agent=agent,
        backend=parent_backend,
        checkpointer=_spec_checkpointer,
        max_turns=max_turns,
        skill_roots=sub_roots,
        memory=spec_mem,
        debug=debug,
        agent_name=an,
        description=spec.get("description", ""),
        interrupt_tools=spec_interrupt,
        hitl_approval_fn=hitl_approval_fn,
    )


def _make_subagent_tool(
    *,
    runner: "DeepAgentRunner",
    tool_name: str,
    checkpointer: str,
    max_turns: int,
    backend: BackendProtocol,
    debug: bool,
) -> Tool:
    agent = runner._agent
    memory_default = runner._memory
    interrupt_tools = runner._interrupt_tools

    async def _invoke(ctx: RunContextWrapper[AgentContext], input: str) -> str:
        si_sub = _skills_prompt_for_backend(backend, runner._skill_roots)
        sub_sid = f"{ctx.context.session_id}:{agent.name}:{uuid.uuid4().hex[:12]}"
        sub_ctx = AgentContext(
            session_id=ctx.context.session_id,
            backend=ctx.context.backend,
            agent_name=agent.name,
            memory=memory_default,
            skills=si_sub,
            debug=debug,
            hitl_tools=interrupt_tools,
            is_subagent=True,
        )
        session = create_session(sub_sid, checkpointer)
        hitl = _make_hitl(
            interrupt_tools, approval_fn=runner._hitl_approval_fn
        )
        wrapped = apply_tool_pipeline(
            list(agent.tools),
            backend,
            agent_name=agent.name,
            debug=debug,
            hitl=hitl,
        )
        agent_wrapped = dataclasses.replace(agent, tools=wrapped)
        result = await Runner.run(
            agent_wrapped,
            input,
            context=sub_ctx,
            session=session,
            hooks=FilesystemHooks(backend),
            max_turns=max_turns,
        )
        return str(result.final_output)

    return function_tool(
        _invoke,
        name_override=tool_name,
        description_override=runner.description,
        use_docstring_info=False,
    )


class DeepAgentRunner:
    def __init__(
        self,
        agent: Agent,
        backend: BackendProtocol,
        checkpointer: str,
        max_turns: int,
        skill_roots: list[str],
        memory: str,
        debug: bool,
        agent_name: str,
        description: str,
        interrupt_tools: list[str],
        hitl_approval_fn: HitlApprovalFn | None = None,
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._checkpointer = checkpointer
        self._max_turns = max_turns
        self._skill_roots = list(skill_roots)
        self._memory = memory
        self._debug = debug
        self._agent_name = agent_name
        self.description = description
        self._interrupt_tools = interrupt_tools
        self._hitl_approval_fn = hitl_approval_fn

    def _make_ctx(self, session_id: str, resume: bool) -> AgentContext:
        si = _skills_prompt_for_backend(self._backend, self._skill_roots)
        return AgentContext(
            session_id=session_id,
            backend=self._backend,
            agent_name=self._agent_name,
            memory=self._memory,
            skills=si,
            debug=self._debug,
            hitl_tools=self._interrupt_tools,
            resume=resume,
        )

    def _make_hooks(self) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        return FilesystemHooks(self._backend)

    def _prepare_agent(self) -> Agent:
        hitl = _make_hitl(
            self._interrupt_tools, approval_fn=self._hitl_approval_fn
        )
        wrapped = apply_tool_pipeline(
            list(self._agent.tools),
            self._backend,
            agent_name=self._agent.name,
            debug=self._debug,
            hitl=hitl,
        )
        return dataclasses.replace(self._agent, tools=wrapped)

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ) -> "DeepRunResult":
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        session = create_session(sid, self._checkpointer)
        agent = self._prepare_agent()
        result = await Runner.run(
            agent,
            task,
            context=ctx,
            session=session,
            hooks=self._make_hooks(),
            max_turns=self._max_turns,
        )
        return DeepRunResult(output=result.final_output, session_id=sid, plan=ctx.plan)

    def run_sync(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ) -> "DeepRunResult":
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
        stream_sink: Callable[[Any], Awaitable[None]] | None = None,
    ):
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        session = create_session(sid, self._checkpointer)
        agent = self._prepare_agent()
        stream: RunResultStreaming = Runner.run_streamed(
            agent,
            task,
            context=ctx,
            session=session,
            hooks=self._make_hooks(),
            max_turns=self._max_turns,
        )
        async for event in stream.stream_events():
            if stream_sink is not None:
                await stream_sink(event)
            yield event

    async def run_with_stream_sink(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
        stream_sink: Callable[[Any], Awaitable[None]],
    ) -> DeepRunResult:
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        session = create_session(sid, self._checkpointer)
        agent = self._prepare_agent()
        stream: RunResultStreaming = Runner.run_streamed(
            agent,
            task,
            context=ctx,
            session=session,
            hooks=self._make_hooks(),
            max_turns=self._max_turns,
        )
        async for event in stream.stream_events():
            await stream_sink(event)
        return DeepRunResult(output=stream.final_output, session_id=sid, plan=ctx.plan)


class DeepRunResult:
    def __init__(self, output: Any, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return (
            f"DeepRunResult(session_id={self.session_id!r}, "
            f"output={str(self.output)[:80]!r})"
        )


def _load_memory(memory: list[str] | None, backend: BackendProtocol) -> str:
    _ = backend
    if not memory:
        return ""
    parts: list[str] = []
    for path in memory:
        p = Path(path).expanduser()
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


DeepAgent = create_deep_agent
