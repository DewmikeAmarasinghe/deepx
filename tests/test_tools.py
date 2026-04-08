"""Unit tests for filesystem and planning tools using InMemoryBackend."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from deepx.backends.memory import InMemoryBackend
from deepx.context import AgentContext
from deepx.models import TodoStatus
from deepx.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from deepx.tools.planning import TodoInput, write_todos


def _make_ctx(session_id: str = "test-session") -> MagicMock:
    backend = InMemoryBackend()
    agent_ctx = AgentContext(
        session_id=session_id,
        backend=backend,
        agent_name="agent",
    )
    ctx = MagicMock()
    ctx.context = agent_ctx
    return ctx


def _call(tool, ctx, **kwargs) -> str:
    return asyncio.run(
        tool.on_invoke_tool._invoke_tool_impl(ctx, json.dumps(kwargs))
    )


class TestLs:
    def test_empty_session(self):
        ctx = _make_ctx()
        result = _call(ls, ctx, path="/")
        assert result == "(empty)"

    def test_lists_files(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "notes.txt", "hi")
        ctx.context.backend.write("test-session", "sub/a.txt", "hello")
        result = _call(ls, ctx, path="/")
        assert "notes.txt" in result
        assert "sub/" in result

    def test_subdirectory_listing(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "sub/a.txt", "a")
        ctx.context.backend.write("test-session", "sub/b.txt", "b")
        result = _call(ls, ctx, path="/sub")
        assert "a.txt" in result
        assert "b.txt" in result


class TestReadFile:
    def test_read_existing_file(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "notes.txt", "line1\nline2\nline3")
        result = _call(read_file, ctx, path="notes.txt", offset=0, limit=100)
        assert "line1" in result
        assert "line2" in result

    def test_read_missing_file(self):
        ctx = _make_ctx()
        result = _call(read_file, ctx, path="missing.txt", offset=0, limit=100)
        assert "Error" in result

    def test_read_with_offset(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "f.txt", "a\nb\nc\nd\ne")
        result = _call(read_file, ctx, path="f.txt", offset=2, limit=2)
        assert "c" in result
        assert "d" in result

    def test_read_empty_file_warning(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "empty.txt", "")
        result = _call(read_file, ctx, path="empty.txt", offset=0, limit=100)
        assert "empty" in result.lower()

    def test_pagination_hint(self):
        ctx = _make_ctx()
        content = "\n".join(str(i) for i in range(150))
        ctx.context.backend.write("test-session", "big.txt", content)
        result = _call(read_file, ctx, path="big.txt", offset=0, limit=100)
        assert "more lines" in result


class TestWriteFile:
    def test_create_new_file(self):
        ctx = _make_ctx()
        result = _call(write_file, ctx, path="new.txt", content="content")
        assert "new.txt" in result
        assert ctx.context.backend.read("test-session", "new.txt") == "content"

    def test_fails_if_exists(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "existing.txt", "old")
        result = _call(write_file, ctx, path="existing.txt", content="new")
        assert "already exists" in result
        assert ctx.context.backend.read("test-session", "existing.txt") == "old"


class TestEditFile:
    def test_simple_replace(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "doc.txt", "hello world")
        result = _call(edit_file, ctx, path="doc.txt", old_string="world", new_string="earth", replace_all=False)
        assert "Successfully replaced" in result
        assert ctx.context.backend.read("test-session", "doc.txt") == "hello earth"

    def test_missing_file_error(self):
        ctx = _make_ctx()
        result = _call(edit_file, ctx, path="missing.txt", old_string="x", new_string="y", replace_all=False)
        assert "Error" in result

    def test_string_not_found_error(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "doc.txt", "hello world")
        result = _call(edit_file, ctx, path="doc.txt", old_string="nope", new_string="yep", replace_all=False)
        assert "Error" in result

    def test_multiple_occurrences_without_replace_all(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "doc.txt", "aa aa aa")
        result = _call(edit_file, ctx, path="doc.txt", old_string="aa", new_string="bb", replace_all=False)
        assert "Error" in result and "3 times" in result

    def test_replace_all(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "doc.txt", "aa aa aa")
        result = _call(edit_file, ctx, path="doc.txt", old_string="aa", new_string="bb", replace_all=True)
        assert "Successfully replaced" in result
        assert ctx.context.backend.read("test-session", "doc.txt") == "bb bb bb"


class TestGlob:
    def test_find_txt_files(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "a.txt", "")
        ctx.context.backend.write("test-session", "b.md", "")
        result = _call(glob, ctx, pattern="*.txt", path="/")
        assert "a.txt" in result
        assert "b.md" not in result

    def test_no_matches(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "a.txt", "")
        result = _call(glob, ctx, pattern="*.py", path="/")
        assert "no matches" in result


class TestGrep:
    def test_find_pattern(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "a.txt", "hello world\nfoo bar")
        ctx.context.backend.write("test-session", "b.txt", "no match here")
        result = _call(grep, ctx, pattern="hello", path=None, glob_pattern=None, output_mode="files_with_matches")
        assert "a.txt" in result
        assert "b.txt" not in result

    def test_content_mode(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "notes.txt", "line1\nhello world\nline3")
        result = _call(grep, ctx, pattern="hello", path=None, glob_pattern=None, output_mode="content")
        assert "hello world" in result

    def test_count_mode(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "notes.txt", "a\na\na")
        result = _call(grep, ctx, pattern="a", path=None, glob_pattern=None, output_mode="count")
        assert "3" in result

    def test_no_matches(self):
        ctx = _make_ctx()
        ctx.context.backend.write("test-session", "a.txt", "nothing here")
        result = _call(grep, ctx, pattern="xyz", path=None, glob_pattern=None, output_mode="files_with_matches")
        assert "no matches" in result


class TestWriteTodos:
    def test_creates_plan(self):
        ctx = _make_ctx()
        result = _call(
            write_todos,
            ctx,
            todos=[
                {"content": "Task 1", "status": "in_progress"},
                {"content": "Task 2", "status": "pending"},
            ],
        )
        assert "Plan saved" in result
        assert len(ctx.context.plan.todos) == 2
        assert ctx.context.plan.todos[0].status == TodoStatus.in_progress

    def test_replaces_existing_plan(self):
        ctx = _make_ctx()
        _call(write_todos, ctx, todos=[{"content": "Old task", "status": "pending"}])
        _call(write_todos, ctx, todos=[{"content": "New task", "status": "in_progress"}])
        assert len(ctx.context.plan.todos) == 1
        assert ctx.context.plan.todos[0].title == "New task"

    def test_persists_plan_to_backend(self):
        ctx = _make_ctx()
        _call(write_todos, ctx, todos=[{"content": "Save me", "status": "pending"}])
        saved = ctx.context.backend.load_plan("test-session", "agent")
        assert saved is not None
        assert "Save me" in saved
