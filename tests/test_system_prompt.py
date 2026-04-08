"""Unit tests for build_system_prompt section ordering and content."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deepx.backends.memory import InMemoryBackend
from deepx.context import AgentContext
from deepx.models import Todo, TodoStatus
from deepx.system_prompt import (
    BASE_AGENT_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    MEMORY_SYSTEM_PROMPT,
    SKILLS_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    TODO_SYSTEM_PROMPT,
    build_system_prompt,
)


def _make_ctx(
    memory: str = "",
    skills_info: str = "",
    hitl_tools: list[str] | None = None,
    todos: list[dict] | None = None,
    files: list[tuple[str, str]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    backend = InMemoryBackend()
    agent_ctx = AgentContext(
        session_id="test",
        backend=backend,
        agent_name="agent",
        memory=memory,
        skills_info=skills_info,
        hitl_tools=hitl_tools or [],
    )
    if todos:
        agent_ctx.plan.todos = [
            Todo(title=t["content"], status=TodoStatus(t.get("status", "pending")))
            for t in todos
        ]
    if files:
        for path, content in files:
            backend.write("test", path, content)

    ctx = MagicMock()
    ctx.context = agent_ctx
    agent = MagicMock()
    agent.name = "agent"
    return ctx, agent


class TestBuildSystemPromptStructure:
    def test_contains_base_prompt(self):
        ctx, agent = _make_ctx()
        prompt = build_system_prompt(ctx, agent)
        assert "Deep Agent" in prompt

    def test_contains_todo_prompt(self):
        ctx, agent = _make_ctx()
        prompt = build_system_prompt(ctx, agent)
        assert "write_todos" in prompt

    def test_contains_filesystem_prompt(self):
        ctx, agent = _make_ctx()
        prompt = build_system_prompt(ctx, agent)
        assert "ls" in prompt
        assert "read_file" in prompt

    def test_contains_task_prompt(self):
        ctx, agent = _make_ctx()
        prompt = build_system_prompt(ctx, agent)
        assert "task" in prompt.lower()

    def test_no_summarization_prompt(self):
        ctx, agent = _make_ctx()
        prompt = build_system_prompt(ctx, agent)
        assert "compact_conversation" not in prompt

    def test_memory_section_only_when_present(self):
        ctx_no_mem, agent = _make_ctx()
        prompt_no_mem = build_system_prompt(ctx_no_mem, agent)
        assert "agent_memory" not in prompt_no_mem

        ctx_with_mem, agent = _make_ctx(memory="I prefer Python.")
        prompt_with_mem = build_system_prompt(ctx_with_mem, agent)
        assert "I prefer Python." in prompt_with_mem
        assert "agent_memory" in prompt_with_mem

    def test_skills_section_only_when_present(self):
        ctx_no_skills, agent = _make_ctx()
        prompt_no_skills = build_system_prompt(ctx_no_skills, agent)
        assert "Skills System" not in prompt_no_skills

        ctx_with_skills, agent = _make_ctx(skills_info="- **web-research**: Research skill")
        prompt_with_skills = build_system_prompt(ctx_with_skills, agent)
        assert "web-research" in prompt_with_skills

    def test_hitl_section_only_when_tools_set(self):
        ctx_no_hitl, agent = _make_ctx()
        prompt_no_hitl = build_system_prompt(ctx_no_hitl, agent)
        assert "Human-in-the-loop" not in prompt_no_hitl

        ctx_hitl, agent = _make_ctx(hitl_tools=["web_search"])
        prompt_hitl = build_system_prompt(ctx_hitl, agent)
        assert "web_search" in prompt_hitl
        assert "Human-in-the-loop" in prompt_hitl

    def test_plan_section_only_when_todos_present(self):
        ctx_no_todos, agent = _make_ctx()
        prompt_no_todos = build_system_prompt(ctx_no_todos, agent)
        assert "Current Plan" not in prompt_no_todos

        ctx_todos, agent = _make_ctx(todos=[{"content": "Do something", "status": "in_progress"}])
        prompt_todos = build_system_prompt(ctx_todos, agent)
        assert "Current Plan" in prompt_todos
        assert "Do something" in prompt_todos

    def test_files_section_only_when_files_present(self):
        ctx_no_files, agent = _make_ctx()
        prompt_no_files = build_system_prompt(ctx_no_files, agent)
        assert "Session Files" not in prompt_no_files

        ctx_files, agent = _make_ctx(files=[("notes.txt", "content")])
        prompt_files = build_system_prompt(ctx_files, agent)
        assert "Session Files" in prompt_files
        assert "notes.txt" in prompt_files

    def test_custom_prompt_prepended(self):
        ctx, agent = _make_ctx()
        prompt = build_system_prompt(ctx, agent, custom_prompt="Custom instructions here.")
        assert prompt.startswith("Custom instructions here.")

    def test_section_order(self):
        ctx, agent = _make_ctx(
            memory="my memory",
            skills_info="- **skill-x**: Skill X",
            hitl_tools=["web_search"],
            todos=[{"content": "task", "status": "pending"}],
            files=[("f.txt", "x")],
        )
        prompt = build_system_prompt(ctx, agent, custom_prompt="Custom")
        positions = {
            "custom": prompt.index("Custom"),
            "base": prompt.index("Deep Agent"),
            "todo": prompt.index("write_todos"),
            "memory": prompt.index("agent_memory"),
            "skills": prompt.index("Skills System"),
            "filesystem": prompt.index("Filesystem Tools"),
            "task": prompt.index("task` (subagent spawner)"),
            "hitl": prompt.index("Human-in-the-loop"),
            "plan": prompt.index("Current Plan"),
            "files": prompt.index("Session Files"),
        }
        order = sorted(positions, key=lambda k: positions[k])
        assert order[0] == "custom"
        assert order[1] == "base"
        assert order[2] == "todo"
        assert order[3] == "memory"
        assert order[4] == "skills"
        assert order[5] == "filesystem"
        assert order[6] == "task"
