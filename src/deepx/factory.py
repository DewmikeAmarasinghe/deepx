"""Deepx agent factory: builds :class:`DeepAgentRunner` on top of the OpenAI Agents SDK.

Layout: constants → subagent ``function_tool`` (nested ``Runner.run`` + per-call session) →
:func:`create_deep_agent` → general-purpose runner → :class:`DeepRunBinding` /
:class:`DeepAgentRunner` → memory loader → exports.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.model_settings import ModelSettings
from agents.result import RunResult, RunResultStreaming
from agents.run_state import RunState
from agents.tool import Tool
from agents.tool_context import ToolContext

from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.utils import resolve_root_dir
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks
from deepx.middleware.hitl import Hitl
from deepx.middleware.logs import SessionToolLogHooks
from deepx.middleware.observability import setup_observability
from deepx.middleware.run_hooks import compose_run_hooks
from deepx.middleware.tool_pipeline import apply_tool_pipeline
from deepx.sessions import create_session
from deepx.system_prompt import (
    build_system_prompt,
    format_skills_for_prompt,
    skills_catalog_for_host,
)
from deepx.tools import builtin_tools_for_backend

DEFAULT_MODEL = "gpt-5-mini"


if TYPE_CHECKING:
    from agents.lifecycle import AgentHooks
    from agents.prompts import DynamicPromptFunction, Prompt

    from deepx.tools.planning import Plan


def _format_subagent_output(run_result: RunResult | RunResultStreaming) -> str:
    """Match ``Agent.as_tool`` output extraction for nested runs."""
    if run_result.final_output is not None and (
        not isinstance(run_result.final_output, str) or run_result.final_output != ""
    ):
        return str(run_result.final_output)

    from agents.items import ItemHelpers, MessageOutputItem, ToolCallOutputItem

    for item in reversed(run_result.new_items):
        if isinstance(item, MessageOutputItem):
            text_output = ItemHelpers.text_message_output(item)
            if text_output:
                return text_output

        if (
            isinstance(item, ToolCallOutputItem)
            and isinstance(item.output, str)
            and item.output
        ):
            return item.output

    out = run_result.final_output
    return "" if out is None else str(out)


# ---------------------------------------------------------------------------
# Subagent wiring
# ---------------------------------------------------------------------------


def _normalize_subagents(
    raw: "Sequence[DeepAgentRunner] | None",
) -> list[DeepAgentRunner]:
    out: list[DeepAgentRunner] = []
    for item in raw or []:
        if isinstance(item, DeepAgentRunner):
            out.append(item)
        else:
            raise TypeError(
                f"subagents entries must be DeepAgentRunner, got {type(item).__name__}"
            )
    return out


def _skills_prompt_for_backend(backend: BackendProtocol, skill_roots: list[str]) -> str:
    host = resolve_root_dir(backend)
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
    return out


def _tool_names_from_tools(tools: Sequence[Tool]) -> set[str]:
    names: set[str] = set()
    for t in tools:
        n = getattr(t, "name", None)
        if isinstance(n, str) and n:
            names.add(n)
    return names


def _validate_interrupt_on_names(
    gated: frozenset[str], tools: Sequence[Tool]
) -> None:
    if not gated:
        return
    known = _tool_names_from_tools(tools)
    unknown = sorted(gated - known)
    if unknown:
        raise ValueError(
            "interrupt_on references unknown tool name(s): "
            + ", ".join(repr(n) for n in unknown)
        )


def _merge_optional_agent_fields(
    *,
    model_settings: ModelSettings | None,
    input_guardrails: list[Any] | None,
    output_guardrails: list[Any] | None,
    agent_hooks: "AgentHooks | None",
    tool_use_behavior: Any | None,
    reset_tool_choice: bool | None,
    prompt: "Prompt | DynamicPromptFunction | None",
) -> dict[str, Any]:
    opts: dict[str, Any] = {}
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
    runner: DeepAgentRunner,
    max_turns: int,
) -> Tool:
    """Expose a subagent as a tool: nested ``Runner.run`` with a per-call session."""
    agent = runner._agent
    tname = re.sub(r"[^a-zA-Z0-9_]", "_", runner._agent_name)

    async def _invoke(ctx: ToolContext, input: str) -> str:
        ac = ctx.context
        if not isinstance(ac, AgentContext):
            raise TypeError("Deepx tools expect AgentContext on the run context")
        sub_backend = runner._backend
        sub_memory = runner._memory
        sub_debug = runner._debug
        ckpt = runner._checkpointer
        skills_sub = _skills_prompt_for_backend(sub_backend, runner._skill_roots)
        sub_sid = f"{ac.session_id}:{agent.name}:{ctx.tool_call_id}"
        sub_ctx = AgentContext(
            session_id=ac.session_id,
            backend=sub_backend,
            agent_name=agent.name,
            memory=sub_memory,
            skills=skills_sub,
            debug=sub_debug,
            resume=False,
            hitl=ac.hitl,
            interrupt_on=runner._interrupt_on,
        )
        session = create_session(sub_sid, ckpt)
        agent_wrapped = runner._prepare_agent()
        hooks = runner._make_hooks()
        run_result = await Runner.run(
            agent_wrapped,
            input,
            context=sub_ctx,
            session=session,
            hooks=hooks,
            max_turns=max_turns,
        )
        return _format_subagent_output(run_result)

    tool = function_tool(
        _invoke,
        name_override=tname,
        description_override=runner.description or "",
        use_docstring_info=False,
    )
    setattr(tool, "_is_agent_tool", True)
    setattr(tool, "_agent_instance", runner._agent)
    return tool


def create_deep_agent(
    model: str = DEFAULT_MODEL,
    tools: list | None = None,
    *,
    name: str = "agent",
    description: str = "",
    system_prompt: str = "",
    subagents: Sequence[DeepAgentRunner] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: type | None = None,
    backend: BackendProtocol | None = None,
    checkpointer: str = ":memory:",
    debug: bool = True,
    max_turns: int = 1000,
    run_hooks: Sequence[RunHooksBase[AgentContext, AgentType[AgentContext]]] = (),
    include_general_purpose: bool = True,
    model_settings: ModelSettings | None = None,
    input_guardrails: list[Any] | None = None,
    output_guardrails: list[Any] | None = None,
    agent_hooks: "AgentHooks | None" = None,
    tool_use_behavior: Any | None = None,
    reset_tool_choice: bool | None = None,
    prompt: "Prompt | DynamicPromptFunction | None" = None,
    interrupt_on: list[str] | None = None,
) -> "DeepAgentRunner":
    """Build a :class:`DeepAgentRunner`: one OpenAI Agents ``Agent`` plus Deepx filesystem, skills, and sessions.

    **Built-ins**

    Flat tool list from :mod:`deepx.tools`: ``write_todos``, ``think_tool``, ``ls``, ``read_file``,
    ``write_file``, ``edit_file``, ``grep``, ``glob``, optional ``execute`` (if the backend
    supports shell), and ``save_memory``. Planning and filesystem behavior are also reflected in
    the dynamic system prompt.

    **Prompting**

    ``system_prompt`` is only the **role / task** section. The rest (skills index, memory, plan
    snapshot, sandbox rules) comes from :func:`deepx.system_prompt.build_system_prompt`.

    ``memory`` is a list of **file paths** only (not inline prose). Files load in list order; body
    text is joined with ``\\n\\n`` before insertion into the **MEMORY** section. Relative paths
    resolve against the backend **host_root** first, then the process **cwd**. ``skills`` roots
    keep their list order in the catalog.

    **Sessions**

    ``checkpointer`` is a SQLite path (including ``\":memory:\"``) passed to
    :func:`deepx.sessions.create_session` when you :meth:`~DeepAgentRunner.bind`. One bound
    conversation uses **one** session store. Nested specialist **tool** calls use a derived session
    id and the child runner's checkpointer.

    **Subagents**

    Each :class:`DeepAgentRunner` is exposed as an SDK ``function_tool`` that runs a nested
    ``Runner.run``. The nested :class:`AgentContext` uses that runner's ``backend``, memory text,
    and debug flag; it **inherits** the parent's ``session_id`` and ``hitl``. Sensitive tools use :data:`interrupt_on` and
    :class:`~deepx.middleware.hitl.Hitl` at :meth:`~DeepAgentRunner.bind` time (not SDK
    ``needs_approval``).

    **Hooks**

    ``run_hooks`` are composed **after** :class:`deepx.middleware.filesystem.FilesystemHooks`.
    ``agent_hooks`` map to ``Agent.hooks`` (per-turn agent callbacks), not run-level hooks.

    **Human-in-the-loop**

    ``interrupt_on`` lists tool names that require host approval before execution. Each non-empty
    string must match a tool on this agent (built-ins + ``tools=`` + subagent tools) or construction
    raises :class:`ValueError`. Pass a :class:`~deepx.middleware.hitl.Hitl` when calling
    :meth:`~DeepAgentRunner.bind`; nested specialist runs inherit the same coordinator from the
    parent context.

    **Remote MCP / Hub tools**

    Deepx does **not** pass ``mcp_servers`` into the OpenAI Agents ``Agent``. Expose MCP servers as
    normal ``function_tool`` / :class:`~agents.tool.FunctionTool` instances on ``tools=`` (for
    example via FastMCP’s ``Client``) so :func:`~deepx.middleware.tool_pipeline.apply_tool_pipeline`
    (eviction + ``interrupt_on`` + :class:`~deepx.middleware.hitl.Hitl`) applies uniformly.

    **Defaults**

    ``model`` is :data:`DEFAULT_MODEL`. ``debug=True`` enables :class:`~deepx.middleware.logs.SessionToolLogHooks`,
    writing tool JSON under ``<data_root>/sessions/<id>/logs/tools`` via the backend runtime API.

    **Returns**

    :class:`DeepAgentRunner`: call :meth:`~DeepAgentRunner.bind` then ``Runner.run`` or
    ``Runner.run_streamed`` on the :class:`DeepRunBinding`.
    """
    setup_observability()

    extra_run_hooks = tuple(run_hooks)
    checkpointer = checkpointer.strip() or ":memory:"
    gated = frozenset(
        name.strip()
        for name in (interrupt_on or [])
        if isinstance(name, str) and name.strip()
    )
    sub_runners = _normalize_subagents(subagents)

    if include_general_purpose and not any(
        r._agent_name == "general_purpose" for r in sub_runners
    ):
        sub_runners = [
            *sub_runners,
            _make_general_purpose_runner(
                model=model,
                user_tools=list(tools or []),
                skills=skills,
                memory=memory,
                resolved_backend=backend or FilesystemBackend(Path.cwd()),
                checkpointer=checkpointer,
                debug=debug,
                max_turns=max_turns,
                extra_run_hooks=extra_run_hooks,
                response_format=response_format,
            ),
        ]

    skill_roots = _collect_skill_roots(skills)
    resolved_backend = backend or FilesystemBackend(Path.cwd())
    mem_content = _load_memory(memory, resolved_backend)
    user_tools = list(tools or [])
    base_tools: list[Tool] = list(builtin_tools_for_backend(backend=resolved_backend))

    subagent_tools: list[Tool] = []
    for sub in sub_runners:
        subagent_tools.append(
            _subagent_tool_from_runner(runner=sub, max_turns=max_turns)
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
        model_settings=model_settings,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        agent_hooks=agent_hooks,
        tool_use_behavior=tool_use_behavior,
        reset_tool_choice=reset_tool_choice,
        prompt=prompt,
    )
    assembled_tools: list[Tool] = base_tools + user_tools + subagent_tools
    _validate_interrupt_on_names(gated, assembled_tools)
    main_agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=assembled_tools,
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
        interrupt_on=gated,
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
    """Same tools/skills/memory file paths as parent; does not add extra handoffs."""
    return create_deep_agent(
        model=model,
        tools=user_tools,
        name="general_purpose",
        description=(
            "General-purpose agent for tasks like, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."
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
        hitl: Hitl | None = None,
    ) -> None:
        self._runner = runner
        self.session_id = session_id
        self.ctx = runner._make_ctx(session_id, resume)
        self.ctx.hitl = hitl
        if hitl is not None:
            hitl.attach_session(runner._backend, session_id)
        self.session = create_session(session_id, runner._checkpointer)
        self.agent = runner._prepare_agent()
        self.hooks = hooks if hooks is not None else runner._make_hooks()

    async def run(self, inp: str | RunState) -> RunResult:
        # Fresh runs wrap ``AgentContext`` in a new :class:`RunContextWrapper` inside the SDK.
        # Resuming from :class:`RunState` must keep ``run_state._context`` (same wrapper as
        # ``result.to_state()``).
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
        """Stream one turn using the bound prepared agent (same tool list as :meth:`run`)."""
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
        interrupt_on: frozenset[str] | None = None,
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
        self._interrupt_on = interrupt_on or frozenset()

    @property
    def backend(self) -> BackendProtocol:
        return self._backend

    def bind(
        self,
        session_id: str,
        *,
        resume: bool = False,
        hooks: RunHooksBase[AgentContext, AgentType[AgentContext]] | None = None,
        hitl: Hitl | None = None,
    ) -> DeepRunBinding:
        return DeepRunBinding(
            self,
            session_id,
            resume=resume,
            hooks=hooks,
            hitl=hitl,
        )

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
            interrupt_on=self._interrupt_on,
        )

    def _make_hooks(self) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        fs = FilesystemHooks(self._backend)
        parts: list[RunHooksBase[AgentContext, AgentType[AgentContext]]] = [fs]
        if self._debug:
            parts.append(SessionToolLogHooks(self._backend))
        parts.extend(self._middleware_hooks)
        return compose_run_hooks(*parts)

    def _prepare_agent(self) -> Agent:
        wrapped = apply_tool_pipeline(
            list(self._agent.tools),
            self._backend,
            interrupt_on=self._interrupt_on,
        )
        return dataclasses.replace(self._agent, tools=wrapped)

    def prepared_agent(self) -> Agent:
        """Agent with filesystem middleware applied (tool pipeline + HITL)."""
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
            return asyncio.run(coro)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return cast("DeepRunResult", pool.submit(asyncio.run, coro).result())

    async def run_stream(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ):
        sid = session_id or uuid.uuid4().hex
        b = self.bind(sid, resume=resume)
        resume_ctx: RunContextWrapper[AgentContext] | AgentContext | None = b.ctx
        stream = Runner.run_streamed(
            b.agent,
            task,
            context=resume_ctx,
            session=b.session,
            hooks=b.hooks,
            max_turns=self._max_turns,
        )
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
    """Load memory files in list order; concatenate with ``\\n\\n`` between files.

    Relative paths try **host_root** first (when the backend has one), then **cwd**.
    ``memory=`` is a ``list[str]`` of file paths only (not inline prose).
    """
    if not memory:
        return ""
    host = resolve_root_dir(backend)
    parts: list[str] = []
    for raw in memory:
        s = (raw or "").strip()
        if not s:
            continue
        p = Path(s).expanduser()
        found: Path | None = None
        if p.is_absolute():
            try:
                rp = p.resolve()
                if rp.is_file():
                    found = rp
            except OSError:
                pass
        else:
            rel = s.lstrip("./")
            candidates: list[Path] = []
            if host is not None:
                candidates.append(host / rel)
            candidates.append(Path.cwd() / s)
            for cand in candidates:
                try:
                    rc = cand.resolve()
                    if rc.is_file():
                        found = rc
                        break
                except OSError:
                    continue
        if found is not None:
            parts.append(found.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


DeepAgent = create_deep_agent

__all__ = [
    "DeepAgent",
    "DeepAgentRunner",
    "DeepRunBinding",
    "DeepRunResult",
    "Hitl",
    "create_deep_agent",
]
