from __future__ import annotations

import json
import tempfile
from pathlib import Path

from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory import InMemoryBackend
from deepx.models import Plan, Todo


def test_in_memory_plan_per_agent() -> None:
    b = InMemoryBackend()
    p1 = Plan(session_id="s1", agent_name="a1", todos=[Todo(title="t1")])
    p2 = Plan(session_id="s1", agent_name="a2", todos=[Todo(title="t2")])
    b.save_plan("s1", "a1", p1.to_json())
    b.save_plan("s1", "a2", p2.to_json())
    assert "t1" in (b.load_plan("s1", "a1") or "")
    assert "t2" in (b.load_plan("s1", "a2") or "")


def test_filesystem_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = FilesystemBackend(root)
        b.write("sid", "research/x.md", "hello")
        assert (root / "sessions/sid/files/research/x.md").read_text() == "hello"
        b.write_store("AGENTS.md", "mem")
        assert (root / "memory/AGENTS.md").read_text() == "mem"
        b.save_plan("sid", "main", Plan(session_id="sid", agent_name="main").to_json())
        assert (root / "sessions/sid/plans/main.json").exists()


def test_append_plan_log_json_array() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        b = FilesystemBackend(Path(tmp))
        b.append_plan_log("s1", json.dumps({"a": 1}))
        b.append_plan_log("s1", json.dumps({"a": 2}))
        p = Path(tmp) / "sessions/s1/logs/plans.json"
        data = json.loads(p.read_text())
        assert len(data) == 2
