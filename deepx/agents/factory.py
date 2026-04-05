from agents import Agent

from deepx.agents.subagent import as_tool
from deepx.instructions import build_instructions
from deepx.middleware.interceptor import ToolInterceptor
from deepx.runner import DeepRunner
from deepx.tools import CORE_TOOLS


def create_agent(
    *,
    model,
    tools=None,
    subagents=None,
    skills_path=None,
    memory_path=None,
    system_prompt="",
    db_path="deepx.db",
    max_turns=200,
    hitl_tools=None,
    hitl_approval_fn=None,
) -> DeepRunner:
    subagent_tools = [as_tool(a, d) for a, d in (subagents or [])]
    all_tools = list(CORE_TOOLS) + subagent_tools + list(tools or [])
    wrapped_tools = ToolInterceptor.apply(all_tools)

    def build_instr(ctx, agent):
        return build_instructions(ctx, agent, user_prompt=system_prompt)

    agent = Agent(
        name="orchestrator",
        instructions=build_instr,
        model=model,
        tools=wrapped_tools,
    )
    return DeepRunner(
        agent,
        db_path,
        max_turns,
        skills_path,
        memory_path,
        hitl_tools,
        hitl_approval_fn,
    )
