"""Deepx agent factory: builds :class:`DeepAgentRunner` on top of the OpenAI Agents SDK.

Layout: constants → :class:`SubagentRef` / helpers → subagent :meth:`agents.Agent.as_tool` wrapper
→ :func:`create_deep_agent` → general-purpose runner → :class:`DeepRunBinding` /
:class:`DeepAgentRunner` → memory loader → exports.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agents import Agent, RunContextWrapper, Runner, handoff
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.model_settings import ModelSettings
from agents.result import RunResult, RunResultStreaming
from agents.run_state import RunState
from agents.tool import Tool

from deepx.backends.filesystem import FilesystemBackend, resolve_host_root
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks, apply_tool_pipeline
from deepx.middleware.observability import setup_observability
from deepx.middleware.run_hooks import compose_run_hooks
from deepx.middleware.subagent_context import SubagentContextIsolationHook
from deepx.sessions import create_session
from deepx.system_prompt import (
    build_system_prompt,
    format_skills_for_prompt,
    skills_catalog_for_host,
)
from deepx.tools import BUILTIN_TOOLS

# Model id for agents 
DEFAULT_MODEL = "gpt-5-mini"

if TYPE_CHECKING:
    from agents.agent import MCPConfig
    from agents.lifecycle import AgentHooks
    from agents.prompts import DynamicPromptFunction, Prompt

    from deepx.tools.planning import Plan


# ---------------------------------------------------------------------------
# Subagent wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubagentRef:
    """Expose another :class:`DeepAgentRunner` as an SDK tool or handoff."""

    runner: "DeepAgentRunner"
    expose: Literal["tool", "handoff"] = "tool"
    needs_approval: bool = False
    tool_name: str | None = None


def _normalize_subagents(
    raw: "Sequence[SubagentRef | DeepAgentRunner] | None",
) -> list[SubagentRef]:
    out: list[SubagentRef] = []
    for item in raw or []:
        if isinstance(item, SubagentRef):
            out.append(item)
        elif isinstance(item, DeepAgentRunner):
            out.append(SubagentRef(item, "tool"))
        else:
            raise TypeError(
                "subagents entries must be SubagentRef or DeepAgentRunner, "
                f"got {type(item).__name__}"
            )
    return out


def _skills_prompt_for_backend(backend: BackendProtocol, skill_roots: list[str]) -> str:
    host = resolve_host_root(backend)
    if host is None:
        return ""
    meta = skills_catalog_for_host(host, skill_roots)
    return format_skills_for_prompt(meta)


def _collect_skill_roots(main: list[str] | None) -> list[str]:
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


def _merge_optional_agent_fields(
    *,
    mcp_servers: list[Any] | None,
    mcp_config: "MCPConfig | None",
    model_settings: ModelSettings | None,
    input_guardrails: list[Any] | None,
    output_guardrails: list[Any] | None,
    agent_hooks: "AgentHooks | None",
    tool_use_behavior: Any | None,
    reset_tool_choice: bool | None,
    prompt: "Prompt | DynamicPromptFunction | None",
) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if mcp_servers is not None:
        opts["mcp_servers"] = mcp_servers
    if mcp_config is not None:
        opts["mcp_config"] = mcp_config
    if model_settings is not None:
        opts["model_settings"] = model_settings
    if input_guardrails is not None:
        opts["input_guardrails"] = input_guardrails
    if output_guardrails is not None:
        opts["output_guardrails"] = output_guardrails
    if agent_hooks is not None:
        opts["hooks"] = agent_hooks
    if tool_use_behavior is not None:
        opts["tool_use_behavior"] = tool_use_behavior
    if reset_tool_choice is not None:
        opts["reset_tool_choice"] = reset_tool_choice
    if prompt is not None:
        opts["prompt"] = prompt
    return opts


def _subagent_tool_from_runner(
    *,
    runner: "DeepAgentRunner",
    tool_name: str,
    max_turns: int,
    backend: BackendProtocol,
    needs_approval: bool,
) -> Tool:
    """Expose a subagent via :meth:`agents.Agent.as_tool` so nested ``needs_approval`` surfaces on the parent run."""
    skills_sub = _skills_prompt_for_backend(backend, runner._skill_roots)
    isolation = SubagentContextIsolationHook(runner._agent_name, skills_sub)
    nested_hooks = compose_run_hooks(
        isolation,
        FilesystemHooks(backend),
        *runner._middleware_hooks,
    )
    return runner.prepared_agent().as_tool(
        tool_name=tool_name,
        tool_description=runner.description or "",
        max_turns=max_turns,
        hooks=nested_hooks,
        needs_approval=needs_approval,
        session=None,
    )


def create_deep_agent(
    model: str = DEFAULT_MODEL,
    tools: list | None = None,
    *,
    name: str = "agent",
    description: str = "",
    system_prompt: str = "",
    subagents: Sequence[SubagentRef | DeepAgentRunner] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: type | None = None,
    backend: BackendProtocol | None = None,
    checkpointer: str = ":memory:",
    debug: bool = True,
    max_turns: int = 1000,
    run_hooks: Sequence[RunHooksBase[AgentContext, AgentType[AgentContext]]] = (),
    include_general_purpose: bool = True,
    mcp_servers: list[Any] | None = None,
    mcp_config: "MCPConfig | None" = None,
    model_settings: ModelSettings | None = None,
    input_guardrails: list[Any] | None = None,
    output_guardrails: list[Any] | None = None,
    agent_hooks: "AgentHooks | None" = None,
    tool_use_behavior: Any | None = None,
    reset_tool_choice: bool | None = None,
    prompt: "Prompt | DynamicPromptFunction | None" = None,
) -> "DeepAgentRunner":
    """Build a Deepx :class:`DeepAgentRunner` around the OpenAI Agents SDK.

    **What Deepx adds**

    - **Built-in tools** (flat list; see :mod:`deepx.tools`): planning (``write_todos``,
      ``think_tool``), filesystem tools (``ls``, ``read_file``, ``write_file``, ``edit_file``,
      ``grep``, ``glob``), optional ``execute`` when the backend supports it, and ``save_memory``.
    - **Dynamic system prompt** via :func:`deepx.system_prompt.build_system_prompt` (skills, memory,
      plan snapshot, filesystem rules). The ``system_prompt=`` argument is the ROLE section only.
    - **Optional SQLite-backed** conversation sessions (``checkpointer`` path or ``:memory:``).
    - **Subagents** as SDK-native tools: each :class:`SubagentRef` with ``expose=\"tool\"`` uses
      :meth:`agents.Agent.as_tool`, so tools inside the specialist that set ``needs_approval`` pause
      the **parent** run and appear in ``RunResult.interruptions`` / streaming interruptions—resolve
      with ``RunState.approve`` / ``reject`` like any other HITL tool. A :class:`SubagentRef` with
      ``expose=\"handoff\"`` registers an SDK ``handoff()`` target; there is no separate
      ``handoffs=`` parameter on this factory.
    - **Run hooks** (``run_hooks``) compose **after** :class:`deepx.middleware.filesystem.FilesystemHooks`
      for every ``Runner.run`` / ``Runner.run_streamed``. **Agent hooks** (``agent_hooks``) map to
      ``Agent.hooks`` and are separate from run hooks.

    **Defaults**

    - ``model`` defaults to :data:`DEFAULT_MODEL` (also used for session compaction in
      :func:`deepx.sessions.create_session`).
    - ``debug=True`` enables tool I/O logging under ``.deepx/sessions/.../logs`` when the backend
      exposes a data root; ``debug=False`` skips those writes.

    **Returns**

    A :class:`DeepAgentRunner` — use :meth:`~DeepAgentRunner.bind` for a :class:`DeepRunBinding`,
    then ``Runner.run`` / ``Runner.run_streamed``. For human-in-the-loop in a host app, drain
    interruptions on the **outer** run and resume with the same agent and session (see SDK HITL docs).
    """
    setup_observability()

    extra_run_hooks = tuple(run_hooks)
    checkpointer = checkpointer.strip() or ":memory:"
    sub_refs = _normalize_subagents(subagents)

    if include_general_purpose and not any(
        r.runner._agent_name == "general_purpose" for r in sub_refs
    ):
        sub_refs = [
            *sub_refs,
            SubagentRef(
                _make_general_purpose_runner(
                    model=model,
                    user_tools=list(tools or []),
                    skills=skills,
                    memory=memory,
                    resolved_backend=backend or FilesystemBackend(Path.home()),
                    checkpointer=checkpointer,
                    debug=debug,
                    max_turns=max_turns,
                    extra_run_hooks=extra_run_hooks,
                    response_format=response_format,
                ),
                "tool",
            ),
        ]

    skill_roots = _collect_skill_roots(skills)
    resolved_backend = backend or FilesystemBackend(Path.home())
    mem_content = _load_memory(memory, resolved_backend)
    user_tools = list(tools or [])
    base_tools: list[Tool] = [*BUILTIN_TOOLS]

    subagent_tools: list[Tool] = []
    subagent_handoffs: list[Any] = []
    for ref in sub_refs:
        runner = ref.runner
        an = runner._agent_name
        tname = ref.tool_name or re.sub(r"[^a-zA-Z0-9_]", "_", an)
        if ref.expose == "handoff":
            subagent_handoffs.append(
                handoff(
                    runner.prepared_agent(),
                    tool_description_override=runner.description or None,
                )
            )
        else:
            subagent_tools.append(
                _subagent_tool_from_runner(
                    runner=runner,
                    tool_name=tname,
                    max_turns=max_turns,
                    backend=resolved_backend,
                    needs_approval=ref.needs_approval,
                )
            )

    _main_checkpointer = checkpointer

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(
            ctx,
            agent,
            custom_prompt=system_prompt,
            checkpointer=_main_checkpointer,
        )

    merged = _merge_optional_agent_fields(
        mcp_servers=mcp_servers,
        mcp_config=mcp_config,
        model_settings=model_settings,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        agent_hooks=agent_hooks,
        tool_use_behavior=tool_use_behavior,
        reset_tool_choice=reset_tool_choice,
        prompt=prompt,
    )
    main_agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=base_tools + user_tools + subagent_tools,
        handoffs=subagent_handoffs,
        output_type=response_format,
        **merged,
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
        middleware_hooks=extra_run_hooks,
    )


def _make_general_purpose_runner(
    *,
    model: str,
    user_tools: list,
    skills: list[str] | None,
    memory: list[str] | None,
    resolved_backend: BackendProtocol,
    checkpointer: str,
    debug: bool,
    max_turns: int,
    extra_run_hooks: tuple[RunHooksBase[AgentContext, AgentType[AgentContext]], ...],
    response_format: type | None,
) -> "DeepAgentRunner":
    """Same tools/skills/memory file paths as parent; does not inherit MCP or extra handoffs."""
    return create_deep_agent(
        model=model,
        tools=user_tools,
        name="general_purpose",
        description=(
            "General-purpose agent for isolated multi-step tasks. "
            "Has access to the same filesystem and planning tools as the main agent."
        ),
        system_prompt="",
        subagents=None,
        skills=list(skills or []),
        memory=memory,
        response_format=response_format,
        backend=resolved_backend,
        checkpointer=checkpointer,
        debug=debug,
        max_turns=max_turns,
        run_hooks=extra_run_hooks,
        include_general_purpose=False,
    )


class DeepRunBinding:
    """Bound agent + SQLite session + hooks for one conversation thread."""

    def __init__(
        self,
        runner: DeepAgentRunner,
        session_id: str,
        *,
        resume: bool,
        hooks: RunHooksBase[AgentContext, AgentType[AgentContext]] | None = None,
    ) -> None:
        self._runner = runner
        self.session_id = session_id
        self.ctx = runner._make_ctx(session_id, resume)
        self.session = create_session(session_id, runner._checkpointer)
        self.agent = runner._prepare_agent()
        self.hooks = hooks if hooks is not None else runner._make_hooks()

    async def run(self, inp: str | RunState) -> RunResult:
        # Fresh runs wrap ``AgentContext`` in a new :class:`RunContextWrapper` inside the SDK.
        # Resuming from :class:`RunState` must keep ``run_state._context`` (same wrapper as
        # ``result.to_state()``) so tool approval decisions in ``_approvals`` are not dropped.
        resume_ctx: RunContextWrapper[AgentContext] | AgentContext | None = (
            None if isinstance(inp, RunState) else self.ctx
        )
        return await Runner.run(
            self.agent,
            inp,
            context=resume_ctx,
            session=self.session,
            hooks=self.hooks,
            max_turns=self._runner._max_turns,
        )

    def run_streamed(self, inp: str | RunState) -> RunResultStreaming:
        resume_ctx: RunContextWrapper[AgentContext] | AgentContext | None = (
            None if isinstance(inp, RunState) else self.ctx
        )
        return Runner.run_streamed(
            self.agent,
            inp,
            context=resume_ctx,
            session=self.session,
            hooks=self.hooks,
            max_turns=self._runner._max_turns,
        )


class DeepAgentRunner:
    """High-level wrapper around an ``Agent`` plus Deepx context and session."""

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
        middleware_hooks: tuple[
            RunHooksBase[AgentContext, AgentType[AgentContext]], ...
        ] = (),
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
        self._middleware_hooks = middleware_hooks

    @property
    def backend(self) -> BackendProtocol:
        return self._backend

    def bind(
        self,
        session_id: str,
        *,
        resume: bool = False,
        hooks: RunHooksBase[AgentContext, AgentType[AgentContext]] | None = None,
    ) -> DeepRunBinding:
        return DeepRunBinding(self, session_id, resume=resume, hooks=hooks)

    def _make_ctx(self, session_id: str, resume: bool) -> AgentContext:
        si = _skills_prompt_for_backend(self._backend, self._skill_roots)
        return AgentContext(
            session_id=session_id,
            backend=self._backend,
            agent_name=self._agent_name,
            memory=self._memory,
            skills=si,
            debug=self._debug,
            resume=resume,
        )

    def _make_hooks(self) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        fs = FilesystemHooks(self._backend)
        if not self._middleware_hooks:
            return fs
        return compose_run_hooks(fs, *self._middleware_hooks)

    def _prepare_agent(self) -> Agent:
        wrapped = apply_tool_pipeline(
            list(self._agent.tools),
            self._backend,
            agent_name=self._agent.name,
            debug=self._debug,
        )
        return dataclasses.replace(self._agent, tools=wrapped)

    def prepared_agent(self) -> Agent:
        """Agent with filesystem middleware applied (for SDK ``handoff``)."""
        return self._prepare_agent()

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
        hooks: RunHooksBase[AgentContext, AgentType[AgentContext]] | None = None,
    ) -> "DeepRunResult":
        sid = session_id or uuid.uuid4().hex
        b = self.bind(sid, resume=resume, hooks=hooks)
        result = await b.run(task)
        return DeepRunResult(
            output=result.final_output,
            session_id=sid,
            plan=b.ctx.plan,
            run_result=result,
        )

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
    ):
        sid = session_id or uuid.uuid4().hex
        b = self.bind(sid, resume=resume)
        stream = b.run_streamed(task)
        async for event in stream.stream_events():
            yield event
        yield {"kind": "done", "output_preview": str(stream.final_output)}


class DeepRunResult:
    def __init__(
        self,
        output: Any,
        session_id: str,
        plan: "Plan",
        *,
        run_result: RunResult | None = None,
    ) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan
        self.run_result = run_result

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

__all__ = [
    "DEFAULT_MODEL",
    "DeepAgent",
    "DeepAgentRunner",
    "DeepRunBinding",
    "DeepRunResult",
    "SubagentRef",
    "create_deep_agent",
]
