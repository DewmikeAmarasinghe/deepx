from agents import Agent, RunContextWrapper, Runner, function_tool

from deepx.context import AgentContext


def as_tool(agent: Agent, description: str):
    async def _run(ctx: RunContextWrapper[AgentContext], task: str) -> str:
        result = await Runner.run(agent, input=task, context=ctx.context)
        path = f"/subagents/{agent.name}_{ctx.context._step_counter:04d}.md"
        ctx.context.vfs[path] = str(result.final_output)
        return str(result.final_output)

    _run.__name__ = f"run_{agent.name}"
    _run.__doc__ = description
    return function_tool(_run)
