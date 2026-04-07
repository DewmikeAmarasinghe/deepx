from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import yaml
from agents import Agent, RunContextWrapper

from deepx.context import AgentContext

BASE_AGENT_PROMPT = """You are an autonomous agent with planning, filesystem tools, memory, and delegation.

Core rules:
- Call write_todos before multi-step work. Update the list as you progress.
- Persist large findings in files under the session filesystem, not only in chat.
- Use read_file before edit_file. Use write_file only for new paths.
- Use update_memory for durable facts that should apply across sessions.
- Use the task tool to delegate isolated subtasks to a specialized sub-agent; you only receive its final output.

Path conventions: paths may be session-rooted (e.g. /research/note.md) or memory store (/store/AGENTS.md)."""


TODO_PROMPT = """## Todos
- write_todos: replace the current todo list with a new ordered list of strings.
- mark_done: mark one item completed by 1-based index.
- read_todos: read todos and statuses."""


FILESYSTEM_PROMPT = """## Filesystem tools
- ls: list files and directories under a path (session or /store/).
- read_file: read with optional line offset and limit; line-numbered output.
- write_file: create a new file; fails if the path already exists.
- edit_file: replace old_string with new_string; optional replace_all.
- append_to_file: append or create.
- glob: match paths with *, **, ?.
- grep: literal text search (not regex); modes files_with_matches, content, count.
- list_files: deprecated alias for ls.
- execute: run shell commands when the backend supports execution."""


TASK_PROMPT = """## Task tool
Call task with subagent_type (registered sub-agent name) and a concise description of work.
You receive only the sub-agent's final output; its intermediate messages stay isolated."""


MEMORY_PROMPT_TEMPLATE = """## Agent memory
<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
Use update_memory to append durable notes. Do not store secrets or credentials.
</memory_guidelines>"""


SKILLS_SYSTEM_PROMPT = """## Skills
Skills use progressive disclosure: only metadata is shown here. When a skill applies, read its SKILL.md with read_file using the path given.

{skills_catalog}"""


HITL_PROMPT = """## Human-in-the-loop
The following tools require explicit human approval before they run in this session: {tools}
"""


class SkillMetadata(TypedDict, total=False):
    name: str
    description: str
    path: str
    license: str | None
    compatibility: str | None
    allowed_tools: list[str]


def discover_skills(paths: list[str]) -> list[SkillMetadata]:
    by_name: dict[str, SkillMetadata] = {}
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = _parse_skill_frontmatter(skill_md.read_text(), str(skill_md.resolve()))
            if meta and meta.get("name") and meta.get("description"):
                by_name[str(meta["name"])] = meta
    return list(by_name.values())


def format_skills_for_prompt(skills: list[SkillMetadata]) -> str:
    if not skills:
        return ""
    lines: list[str] = []
    for skill in skills:
        lines.append(f"- **{skill['name']}**: {skill['description']}")
        lines.append(f"  -> Read `{skill['path']}` for full instructions")
    return "\n".join(lines)


def _parse_skill_frontmatter(content: str, path: str) -> SkillMetadata | None:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()
    if not name or not description:
        return None
    meta: SkillMetadata = {
        "name": name,
        "description": description,
        "path": path,
    }
    if data.get("license") is not None:
        meta["license"] = str(data.get("license"))
    if data.get("compatibility") is not None:
        meta["compatibility"] = str(data.get("compatibility"))
    at = data.get("allowed-tools") or data.get("allowed_tools")
    if isinstance(at, list):
        meta["allowed_tools"] = [str(x) for x in at]
    elif isinstance(at, str):
        meta["allowed_tools"] = [at]
    return meta


def build_system_prompt(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    custom_prompt: str = "",
) -> str:
    sections: list[str] = []

    if custom_prompt:
        sections.append(custom_prompt)

    sections.append(BASE_AGENT_PROMPT)
    sections.append(TODO_PROMPT)

    if ctx.context.memory:
        sections.append(
            MEMORY_PROMPT_TEMPLATE.format(agent_memory=ctx.context.memory)
        )

    if ctx.context.skills_info:
        sections.append(
            SKILLS_SYSTEM_PROMPT.format(skills_catalog=ctx.context.skills_info)
        )

    sections.append(FILESYSTEM_PROMPT)
    sections.append(TASK_PROMPT)

    if ctx.context.hitl_tools:
        sections.append(
            HITL_PROMPT.format(tools=", ".join(ctx.context.hitl_tools))
        )

    if ctx.context.plan.todos:
        lines = [
            f"[{i + 1}] ({t.status.value}) {t.title}"
            for i, t in enumerate(ctx.context.plan.todos)
        ]
        sections.append("## Current Plan\n" + "\n".join(lines))

    files = ctx.context.backend.list_files(ctx.context.session_id)
    if files:
        shown = files[:50]
        block = "\n".join(shown)
        if len(files) > 50:
            block += f"\n... and {len(files) - 50} more. Use ls with a prefix to filter."
        sections.append(f"## Workspace Files\n{block}")

    return "\n\n---\n\n".join(sections)
