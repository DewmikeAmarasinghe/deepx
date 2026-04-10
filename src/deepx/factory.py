from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from pathlib import Path
from typing import Any

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

# Public type alias for dict-based subagent specs.
# Supported keys mirror the parameters of create_deep_agent.
SubAgentDict = dict


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
) -> "DeepAgentRunner":
    setup_observability()

    checkpointer = checkpointer.strip() or ":memory:"
    resolved_backend = backend or FilesystemBackend(root_dir=".deepx")
    mem_content = _load_memory(memory, resolved_backend)
    skills_paths = skills or []
    skills_info_main = format_skills_for_prompt(discover_skills(skills_paths))
    user_tools = list(tools or [])
    base_tools = [*FILESYSTEM_TOOLS, *PLANNING_TOOLS]

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
                "tools": user_tools,
                "skills": skills_paths,
            }
        )

    hitl = HumanInTheLoopHooks(interrupt_on) if interrupt_on else None
    interrupt_list = list(interrupt_on or [])

    subagent_tools: list[Tool] = []
    for spec in sub_specs:
        runner = _resolve_subagent_spec(
            spec,
            parent_model=model,
            parent_backend=resolved_backend,
            checkpointer=checkpointer,
            parent_hitl=hitl,
            debug=debug,
            mem_content=mem_content,
            skills_paths=skills_paths,
            user_tools=user_tools,
            parent_interrupt=interrupt_list,
            max_turns=max_turns,
            parent_response_format=response_format,
        )
        an = runner._agent_name
        tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", an) + "_subagent"
        subagent_tools.append(
            _make_subagent_tool(
                runner=runner,
                tool_name=tool_name,
                checkpointer=checkpointer,
                max_turns=max_turns,
                backend=resolved_backend,
                hitl=hitl,
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
        hitl=hitl,
        skills_info=skills_info_main,
        memory=mem_content,
        debug=debug,
        agent_name=name,
        description=description,
        interrupt_tools=interrupt_list,
    )


def _resolve_subagent_spec(
    spec: dict | "DeepAgentRunner",
    *,
    parent_model: str,
    parent_backend: BackendProtocol,
    checkpointer: str,
    parent_hitl: HumanInTheLoopHooks | None,
    debug: bool,
    mem_content: str,
    skills_paths: list[str],
    user_tools: list,
    parent_interrupt: list[str],
    max_turns: int,
    parent_response_format: type | None,
) -> "DeepAgentRunner":
    """Normalise a subagent spec (dict or existing DeepAgentRunner) into a DeepAgentRunner.

    Builds the runner directly to avoid the top-level auto-add of general_purpose
    that would otherwise cause infinite recursion.
    """
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

    skills_info = format_skills_for_prompt(discover_skills(spec_skills))
    spec_mem = _load_memory(spec_memory_paths, parent_backend) if spec_memory_paths else mem_content if is_gp else ""
    spec_hitl = HumanInTheLoopHooks(spec_interrupt) if spec_interrupt else parent_hitl

    base_tools = [*FILESYSTEM_TOOLS, *PLANNING_TOOLS]

    # Recursively build nested subagent tools if the spec defines its own subagents
    nested_subagent_tools: list[Tool] = []
    for nested in spec.get("subagents") or []:
        nested_runner = _resolve_subagent_spec(
            nested,
            parent_model=spec_model,
            parent_backend=parent_backend,
            checkpointer=checkpointer,
            parent_hitl=spec_hitl,
            debug=debug,
            mem_content=spec_mem,
            skills_paths=spec_skills,
            user_tools=spec_user_tools,
            parent_interrupt=spec_interrupt,
            max_turns=max_turns,
            parent_response_format=spec_response_format,
        )
        nested_tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", nested_runner._agent_name) + "_subagent"
        nested_subagent_tools.append(
            _make_subagent_tool(
                runner=nested_runner,
                tool_name=nested_tool_name,
                checkpointer=checkpointer,
                max_turns=max_turns,
                backend=parent_backend,
                hitl=spec_hitl,
                debug=debug,
            )
        )

    custom_prompt = spec.get("system_prompt", "")
    _spec_checkpointer = checkpointer

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
        checkpointer=checkpointer,
        max_turns=max_turns,
        hitl=spec_hitl,
        skills_info=skills_info,
        memory=spec_mem,
        debug=debug,
        agent_name=an,
        description=spec.get("description", ""),
        interrupt_tools=spec_interrupt,
    )


def _make_subagent_tool(
    *,
    runner: "DeepAgentRunner",
    tool_name: str,
    checkpointer: str,
    max_turns: int,
    backend: BackendProtocol,
    hitl: HumanInTheLoopHooks | None,
    debug: bool,
) -> Tool:
    """Create a named FunctionTool that runs a subagent in an isolated context."""
    agent = runner._agent
    skills_info = runner._skills_info
    memory_default = runner._memory
    interrupt_tools = runner._interrupt_tools

    async def _invoke(ctx: RunContextWrapper[AgentContext], input: str) -> str:
        sub_ctx = AgentContext(
            session_id=ctx.context.session_id,
            backend=ctx.context.backend,
            agent_name=agent.name,
            memory=memory_default,
            skills_info=skills_info,
            debug=debug,
            hitl_tools=interrupt_tools,
            is_subagent=True,
        )
        sub_sid = f"{ctx.context.session_id}:{agent.name}:{uuid.uuid4().hex[:12]}"
        session = create_session(sub_sid, checkpointer)
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
        hitl: HumanInTheLoopHooks | None,
        skills_info: str,
        memory: str,
        debug: bool,
        agent_name: str,
        description: str,
        interrupt_tools: list[str],
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._checkpointer = checkpointer
        self._max_turns = max_turns
        self._hitl = hitl
        self._skills_info = skills_info
        self._memory = memory
        self._debug = debug
        self._agent_name = agent_name
        self.description = description
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

    def _prepare_agent(self) -> Agent:
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
    ) -> "DeepRunResult":
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        session = create_session(sid, self._checkpointer)
        agent = self._prepare_agent()
        result = await Runner.run(
            agent,
            input=task,
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
        """Run one agent turn. For multi-turn CLIs, prefer a single event loop + `run()` (see deepx_cli.session)."""
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
        session = create_session(sid, self._checkpointer)
        agent = self._prepare_agent()
        stream: RunResultStreaming = Runner.run_streamed(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=self._make_hooks(),
            max_turns=self._max_turns,
        )
        async for event in stream.stream_events():
            yield event


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
