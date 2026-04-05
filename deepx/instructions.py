from agents import Agent, RunContextWrapper

from deepx.context import AgentContext

BASE_PROMPT = (
    "You are a deep autonomous agent capable of long-running, multi-step tasks.\n\n"
    "## behavior rules\n"
    "- call write_todos FIRST before any multi-step task — commit to a plan before acting\n"
    "- write findings to files using write_file — never hold large content only in conversation\n"
    "- pass file paths to subagents, not raw content — "
    "subagents can read from the shared workspace\n"
    "- call update_memory for any fact you will need across sessions or restarts\n"
    "- when tasks are independent, call multiple subagent tools in the same response — "
    "they run in parallel\n"
    "- after major steps, verify output before reporting done\n"
    "- if a tool output was evicted to /outputs/, use read_file to access it\n\n"
    "## workspace file convention\n"
    "/plan.md              current task plan (written by write_todos)\n"
    "/memory/notes.md      persistent memory (written by update_memory)\n"
    "/research/*.md        web scraping and external data\n"
    "/db/*.md              sql query results\n"
    "/subagents/*.md       subagent outputs (auto-written when subagents complete)\n"
    "/outputs/*.md         auto-saved large tool results (too large to display inline)\n"
    "/obs/*.json           observability log — every tool call recorded here, do not edit\n"
)


def build_instructions(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    user_prompt: str = "",
) -> str:
    parts = [BASE_PROMPT]
    if user_prompt.strip():
        parts.append(user_prompt.strip())
    if ctx.context.todos:
        parts.append("## current todos\n" + "\n".join(f"- {t}" for t in ctx.context.todos))
    if ctx.context.vfs:
        paths = sorted(ctx.context.vfs.keys())[:40]
        parts.append("## workspace files\n" + "\n".join(paths))
    if ctx.context.memory.strip():
        parts.append("## memory\n" + ctx.context.memory.strip())
    if ctx.context.skills_info.strip():
        parts.append("## skills\n" + ctx.context.skills_info.strip())
    if ctx.context.visited_urls:
        urls = sorted(ctx.context.visited_urls)[:20]
        parts.append("## visited urls\n" + "\n".join(urls))
    return "\n\n".join(parts)
