"""Unit tests for FilesystemBackend, InMemoryBackend, and CompositeBackend."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from deepx.backends.composite import CompositeBackend
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory import InMemoryBackend


class TestInMemoryBackend:
    def setup_method(self):
        self.backend = InMemoryBackend()

    def test_write_and_read(self):
        self.backend.write("s1", "notes.txt", "hello")
        assert self.backend.read("s1", "notes.txt") == "hello"

    def test_read_missing_returns_none(self):
        assert self.backend.read("s1", "missing.txt") is None

    def test_exists(self):
        self.backend.write("s1", "a.txt", "x")
        assert self.backend.exists("s1", "a.txt")
        assert not self.backend.exists("s1", "b.txt")

    def test_append(self):
        self.backend.write("s1", "log.txt", "line1\n")
        self.backend.append("s1", "log.txt", "line2\n")
        assert self.backend.read("s1", "log.txt") == "line1\nline2\n"

    def test_list_files(self):
        self.backend.write("s1", "a.txt", "a")
        self.backend.write("s1", "sub/b.txt", "b")
        files = self.backend.list_files("s1")
        assert "a.txt" in files
        assert "sub/b.txt" in files

    def test_list_files_with_prefix(self):
        self.backend.write("s1", "sub/a.txt", "a")
        self.backend.write("s1", "other/b.txt", "b")
        files = self.backend.list_files("s1", "sub/")
        assert "sub/a.txt" in files
        assert "other/b.txt" not in files

    def test_store_read_write(self):
        self.backend.write_store("AGENTS.md", "memory content")
        assert self.backend.read_store("AGENTS.md") == "memory content"

    def test_store_list(self):
        self.backend.write_store("a.md", "x")
        self.backend.write_store("b.md", "y")
        stored = self.backend.list_store()
        assert "a.md" in stored
        assert "b.md" in stored

    def test_save_and_load_plan(self):
        plan_json = '{"agent_name": "agent", "todos": []}'
        self.backend.save_plan("s1", "agent", plan_json)
        loaded = self.backend.load_plan("s1", "agent")
        assert loaded == plan_json

    def test_load_plan_missing_returns_none(self):
        assert self.backend.load_plan("s1", "nonexistent") is None

    def test_per_agent_plan_isolation(self):
        self.backend.save_plan("s1", "agent-a", '{"agent_name": "a"}')
        self.backend.save_plan("s1", "agent-b", '{"agent_name": "b"}')
        assert self.backend.load_plan("s1", "agent-a") == '{"agent_name": "a"}'
        assert self.backend.load_plan("s1", "agent-b") == '{"agent_name": "b"}'

    def test_per_session_plan_isolation(self):
        self.backend.save_plan("s1", "agent", '{"s": "1"}')
        self.backend.save_plan("s2", "agent", '{"s": "2"}')
        assert self.backend.load_plan("s1", "agent") == '{"s": "1"}'
        assert self.backend.load_plan("s2", "agent") == '{"s": "2"}'

    def test_append_task_log(self):
        self.backend.append_task_log("s1", "do something")
        self.backend.append_task_log("s1", "do another thing")

    def test_append_plan_log(self):
        self.backend.append_plan_log("s1", '{"action": "update"}')

    def test_save_tool_log(self):
        self.backend.save_tool_log("s1", {"tool": "ls", "output": "files..."})

    def test_supports_execution_false(self):
        assert self.backend.supports_execution is False


class TestFilesystemBackend:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        self.backend = FilesystemBackend(self.root)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_write_and_read(self):
        self.backend.write("s1", "notes.txt", "hello world")
        assert self.backend.read("s1", "notes.txt") == "hello world"

    def test_read_missing_returns_none(self):
        assert self.backend.read("s1", "missing.txt") is None

    def test_path_layout(self):
        self.backend.write("s1", "output/report.md", "# Report")
        expected = Path(self.root) / "sessions" / "s1" / "files" / "output" / "report.md"
        assert expected.exists()
        assert expected.read_text() == "# Report"

    def test_list_files(self):
        self.backend.write("s1", "a.txt", "a")
        self.backend.write("s1", "sub/b.txt", "b")
        files = self.backend.list_files("s1")
        assert "a.txt" in files
        assert "sub/b.txt" in files

    def test_store_path_layout(self):
        self.backend.write_store("AGENTS.md", "# Memory")
        expected = Path(self.root) / "memory" / "AGENTS.md"
        assert expected.exists()

    def test_save_plan_path_layout(self):
        self.backend.save_plan("s1", "my-agent", '{"todos": []}')
        plan_dir = Path(self.root) / "sessions" / "s1" / "plans"
        assert any(plan_dir.iterdir())

    def test_load_plan_roundtrip(self):
        data = '{"agent_name": "x", "todos": []}'
        self.backend.save_plan("s1", "x", data)
        assert self.backend.load_plan("s1", "x") == data

    def test_append_plan_log(self):
        self.backend.append_plan_log("s1", '{"step": 1}')
        self.backend.append_plan_log("s1", '{"step": 2}')
        log_dir = Path(self.root) / "sessions" / "s1" / "logs"
        assert log_dir.exists()
        log_files = list(log_dir.iterdir())
        assert log_files, "Expected at least one log file in logs/"

    def test_exists(self):
        self.backend.write("s1", "file.txt", "x")
        assert self.backend.exists("s1", "file.txt")
        assert not self.backend.exists("s1", "other.txt")

    def test_append(self):
        self.backend.write("s1", "log.txt", "line1\n")
        self.backend.append("s1", "log.txt", "line2\n")
        assert self.backend.read("s1", "log.txt") == "line1\nline2\n"

    def test_append_system_prompt_log(self):
        import json
        self.backend.append_system_prompt_log("s1", "orchestrator", "You are an agent.")
        self.backend.append_system_prompt_log("s1", "orchestrator", "You are an agent. (updated)")
        log_path = Path(self.root) / "sessions" / "s1" / "logs" / "system_prompts.json"
        assert log_path.exists()
        entries = json.loads(log_path.read_text())
        assert len(entries) == 2
        assert entries[0]["agent"] == "orchestrator"
        assert "You are an agent." in entries[0]["prompt"]


class TestCompositeBackend:
    def setup_method(self):
        self.default_backend = InMemoryBackend()
        self.session_backend = InMemoryBackend()
        self.backend = CompositeBackend(
            default=self.default_backend,
            routes={"sessions/": self.session_backend},
        )

    def test_write_routes_to_default(self):
        self.backend.write("s1", "file.txt", "content")
        assert self.default_backend.read("s1", "file.txt") == "content"

    def test_write_routes_via_prefix(self):
        self.backend.write("s1", "sessions/custom.txt", "content")
        assert self.session_backend.read("s1", "custom.txt") == "content"
        assert self.default_backend.read("s1", "sessions/custom.txt") is None

    def test_store_goes_to_default(self):
        self.backend.write_store("key.md", "value")
        assert self.default_backend.read_store("key.md") == "value"

    def test_read_routes_to_default(self):
        self.default_backend.write("s1", "notes.txt", "hi")
        assert self.backend.read("s1", "notes.txt") == "hi"

    def test_plan_delegated_to_default(self):
        self.backend.save_plan("s1", "agent", '{"todos": []}')
        loaded = self.backend.load_plan("s1", "agent")
        assert loaded is not None
