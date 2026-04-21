"""Deepx agent factory: builds :class:`DeepAgentRunner` on top of the OpenAI Agents SDK.

Layout: constants → :class:`SubagentRef` / helpers → subagent ``function_tool`` (nested
``Runner.run`` + per-call session) → :func:`create_deep_agent` → general-purpose runner →
:class:`DeepRunBinding` / :class:`DeepAgentRunner` → memory loader → exports.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from agents import Agent, RunContextWrapper, Runner, function_tool, handoff
from agents._tool_identity import get_function_tool_approval_keys
from agents.agent import Agent as AgentType
from agents.agent_tool_state import (
    consume_agent_tool_run_result,
    get_agent_tool_state_scope,
    peek_agent_tool_run_result,
    record_agent_tool_run_result,
    set_agent_tool_state_scope,
)
from agents.items import ToolApprovalItem
from agents.lifecycle import RunHooksBase
from agents.model_settings import ModelSettings
from agents.result import RunResult, RunResultStreaming
from agents.run_state import RunState
from agents.tool import Tool
from agents.tool_context import ToolContext

from deepx.backends.filesystem import FilesystemBackend, resolve_host_root
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks, apply_tool_pipeline
from deepx.middleware.observability import setup_observability
from deepx.middleware.run_hooks import compose_run_hooks
from deepx.sessions import create_session
from deepx.system_prompt import (
    build_system_prompt,
    format_skills_for_prompt,
    skills_catalog_for_host,
)
from deepx.tools import BUILTIN_TOOLS

DEFAULT_MODEL = "gpt-5-mini"

if TYPE_CHECKING:
    from agents.agent import MCPConfig
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
    checkpointer: str,
    max_turns: int,
    backend: BackendProtocol,
    needs_approval: bool,
    temporal_workflow: bool,
) -> Tool:
    """Expose a subagent as a tool: nested ``Runner.run`` with SDK-aligned agent-tool HITL.

    Nested ``needs_approval`` tools surface on the parent run via
    ``record_agent_tool_run_result`` / ``peek_agent_tool_run_result``. Each delegation uses its own
    session (SQLite path or async list when ``temporal_workflow``).
    """
    agent = runner._agent

    async def _invoke(ctx: ToolContext, input: str) -> str:
        ac = ctx.context
        if not isinstance(ac, AgentContext):
            raise TypeError("Deepx tools expect AgentContext on the run context")
        skills_sub = _skills_prompt_for_backend(backend, runner._skill_roots)
        sub_sid = f"{ac.session_id}:{agent.name}:{ctx.tool_call_id}"
        sub_ctx = AgentContext(
            session_id=ac.session_id,
            backend=ac.backend,
            agent_name=agent.name,
            memory=runner._memory,
            skills=skills_sub,
            debug=runner._debug,
            resume=False,
            is_subagent=True,
        )
        session = create_session(
            sub_sid, checkpointer, temporal_workflow=temporal_workflow
        )
        agent_wrapped = runner._prepare_agent()
        hooks = runner._make_hooks()
        tool_state_scope_id = get_agent_tool_state_scope(ctx)

        nested_context = ToolContext(
            context=sub_ctx,
            usage=ctx.usage,
            tool_name=ctx.tool_name,
            tool_call_id=ctx.tool_call_id,
            tool_arguments=ctx.tool_arguments,
            tool_call=ctx.tool_call,
            tool_namespace=ctx.tool_namespace,
            agent=runner._agent,
            run_config=ctx.run_config,
            turn_input=list(ctx.turn_input),
            _approvals=dict(ctx._approvals),
        )
        set_agent_tool_state_scope(nested_context, tool_state_scope_id)

        def _nested_approvals_status(
            interruptions: list[ToolApprovalItem],
        ) -> Literal["approved", "pending", "rejected"]:
            has_pending = False
            has_decision = False
            for interruption in interruptions:
                call_id = interruption.call_id
                if not call_id:
                    has_pending = True
                    continue
                tool_namespace = RunContextWrapper._resolve_tool_namespace(interruption)
                status = ctx.get_approval_status(
                    interruption.tool_name or "",
                    call_id,
                    tool_namespace=tool_namespace,
                    existing_pending=interruption,
                )
                if status is False:
                    return "rejected"
                if status is True:
                    has_decision = True
                if status is None:
                    has_pending = True
            if has_decision:
                return "approved"
            if has_pending:
                return "pending"
            return "approved"

        def _apply_nested_approvals(
            nested_run_ctx: RunContextWrapper[Any],
            parent_context: RunContextWrapper[Any],
            interruptions: list[ToolApprovalItem],
        ) -> None:
            def _find_mirrored_approval_record(
                interruption: ToolApprovalItem,
                *,
                approved: bool,
            ) -> Any | None:
                candidate_keys = list(
                    RunContextWrapper._resolve_approval_keys(interruption)
                )
                for candidate_key in get_function_tool_approval_keys(
                    tool_name=RunContextWrapper._resolve_tool_name(interruption),
                    tool_namespace=RunContextWrapper._resolve_tool_namespace(
                        interruption
                    ),
                    tool_lookup_key=RunContextWrapper._resolve_tool_lookup_key(
                        interruption
                    ),
                    include_legacy_deferred_key=True,
                ):
                    if candidate_key not in candidate_keys:
                        candidate_keys.append(candidate_key)
                fallback: Any | None = None
                for candidate_key in candidate_keys:
                    candidate = parent_context._approvals.get(candidate_key)
                    if candidate is None:
                        continue
                    if approved and candidate.approved is True:
                        return candidate
                    if not approved and candidate.rejected is True:
                        return candidate
                    if fallback is None:
                        fallback = candidate
                return fallback

            for interruption in interruptions:
                call_id = interruption.call_id
                if not call_id:
                    continue
                tool_name_i = RunContextWrapper._resolve_tool_name(interruption)
                tool_namespace = RunContextWrapper._resolve_tool_namespace(interruption)
                approval_key = RunContextWrapper._resolve_approval_key(interruption)
                status = parent_context.get_approval_status(
                    tool_name_i,
                    call_id,
                    tool_namespace=tool_namespace,
                    existing_pending=interruption,
                )
                if status is None:
                    continue
                approval_record = parent_context._approvals.get(approval_key)
                if approval_record is None:
                    approval_record = _find_mirrored_approval_record(
                        interruption,
                        approved=status,
                    )
                if status is True:
                    always_approve = bool(
                        approval_record and approval_record.approved is True
                    )
                    nested_run_ctx.approve_tool(
                        interruption,
                        always_approve=always_approve,
                    )
                else:
                    always_reject = bool(
                        approval_record and approval_record.rejected is True
                    )
                    nested_run_ctx.reject_tool(
                        interruption,
                        always_reject=always_reject,
                    )

        run_result: RunResult | RunResultStreaming | None = None
        resume_state: RunState | None = None
        should_record_run_result = True

        if ctx.tool_call is not None:
            pending_run_result = peek_agent_tool_run_result(
                ctx.tool_call,
                scope_id=tool_state_scope_id,
            )
            if pending_run_result and getattr(
                pending_run_result, "interruptions", None
            ):
                status = _nested_approvals_status(pending_run_result.interruptions)
                if status == "pending":
                    run_result = pending_run_result
                    should_record_run_result = False
                elif status in ("approved", "rejected"):
                    resume_state = pending_run_result.to_state()
                    if resume_state._context is not None:
                        _apply_nested_approvals(
                            resume_state._context,
                            ctx,
                            pending_run_result.interruptions,
                        )
                    consume_agent_tool_run_result(
                        ctx.tool_call,
                        scope_id=tool_state_scope_id,
                    )

        if run_result is None:
            run_result = await Runner.run(
                agent_wrapped,
                resume_state if resume_state is not None else input,
                context=None if resume_state is not None else cast(Any, nested_context),
                session=session,
                hooks=hooks,
                max_turns=max_turns,
            )

        interruptions = getattr(run_result, "interruptions", None)
        if ctx.tool_call is not None and interruptions and should_record_run_result:
            record_agent_tool_run_result(
                ctx.tool_call,
                run_result,
                scope_id=tool_state_scope_id,
            )

        return _format_subagent_output(run_result)

    tool = function_tool(
        _invoke,
        name_override=tool_name,
        description_override=runner.description or "",
        use_docstring_info=False,
        needs_approval=needs_approval,
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
    temporal_workflow: bool = False,
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

    **Sessions**

    ``checkpointer`` is the SQLite path (or ``:memory:``) passed to :func:`deepx.sessions.create_session`
    when you :meth:`~DeepAgentRunner.bind`. One bound conversation uses **one** session database;
    a specialist reached via **handoff** shares that same run and session—its own ``checkpointer``
    on the runner is relevant when that specialist is **bound and run alone**, not for in-run
    handoff history.

    **Subagents (:class:`SubagentRef`)**

    - ``expose=\"tool\"`` (default): specialist exposed with a ``function_tool`` that runs a nested
      ``Runner.run`` and a **per-call** session derived from the parent ``session_id``, subagent
      name, and ``tool_call_id``. Nested tools with ``needs_approval`` surface as interruptions on
      the **parent** run; use ``RunState.approve`` / ``reject``.
    - ``expose=\"handoff\"``: specialist registered with the SDK ``handoff()`` helper on the main
      ``Agent`` (there is no separate ``handoffs=`` argument on this factory). Use for long-lived
      context switches (e.g. SQL specialist) where the model should **transfer** control rather than
      return through a tool result.

    **Hooks**

    ``run_hooks`` are composed **after** :class:`deepx.middleware.filesystem.FilesystemHooks`.
    ``agent_hooks`` map to ``Agent.hooks`` (per-turn agent callbacks), not run-level hooks.

    **Temporal (demo)**

    For fine-grained Temporal history (plugin-managed activities per model/tool step), await
    ``Runner.run`` / ``run_binding_until_settled`` from **workflow** code with
    ``OpenAIAgentsPlugin`` and ``UnsandboxedWorkflowRunner``—see ``test_demo/temporal/``.

    **Defaults**

    ``model`` is :data:`DEFAULT_MODEL`. ``debug=True`` writes tool JSON logs under
    ``.deepx/sessions/<id>/logs`` when the backend exposes a workspace root.

    **Returns**

    :class:`DeepAgentRunner`: call :meth:`~DeepAgentRunner.bind` then ``Runner.run`` or
    ``Runner.run_streamed`` on the :class:`DeepRunBinding`.
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
                    temporal_workflow=temporal_workflow,
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
                    checkpointer=runner._checkpointer,
                    max_turns=max_turns,
                    backend=resolved_backend,
                    needs_approval=ref.needs_approval,
                    temporal_workflow=temporal_workflow,
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
        temporal_workflow=temporal_workflow,
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
    temporal_workflow: bool,
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
        temporal_workflow=temporal_workflow,
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
        self.session = create_session(
            session_id,
            runner._checkpointer,
            temporal_workflow=runner._temporal_workflow,
        )
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
        temporal_workflow: bool = False,
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
        self._temporal_workflow = temporal_workflow

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
    "DeepAgent",
    "DeepAgentRunner",
    "DeepRunBinding",
    "DeepRunResult",
    "SubagentRef",
    "create_deep_agent",
]
