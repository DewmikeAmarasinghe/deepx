from __future__ import annotations
import uuid
from datetime import datetime, timezone
from agents import RunHooks, RunContextWrapper, Agent
from agents.run_context import AgentHookContext
from deepx.context import AgentContext
from deepx.middleware._utils import LARGE_OUTPUT_THRESHOLD, generate_preview, sanitize_path_component
from deepx.models import Plan


class WorkspaceHooks(RunHooks[AgentContext]):
    def __init__(self, backend) -> None:
        self._backend = backend

    async def on_agent_start(
        self, ctx: AgentHookContext[AgentContext], agent: Agent
    ) -> None:
        if not ctx.context.memory:
            raw = self._backend.read_shared("AGENTS.md")
            if raw:
                ctx.context.memory = raw

        saved_plan = self._backend.load_plan(ctx.context.session_id)
        if saved_plan:
            ctx.context.plan = Plan.model_validate_json(saved_plan)

    async def on_tool_end(
        self, ctx: RunContextWrapper[AgentContext], agent: Agent, tool, result: str
    ) -> None:
        call_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()
        session_id = ctx.context.session_id
        tool_name = sanitize_path_component(tool.name)
        is_large = len(result) > LARGE_OUTPUT_THRESHOLD
        saved_to = None

        if is_large:
            file_path = f"{tool_name}_{call_id}.txt"
            self._backend.write(session_id, file_path, result)
            saved_to = file_path

        log = {
            "call_id": call_id,
            "tool_name": tool.name,
            "agent_name": agent.name,
            "session_id": session_id,
            "timestamp": timestamp,
            "output_chars": len(result),
            "output_preview": generate_preview(result, 10),
            "saved_to": saved_to,
        }
        self._backend.save_tool_log(session_id, log)