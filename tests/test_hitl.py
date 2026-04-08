"""Unit tests for HITL approval via tool invoke wrapping."""
from __future__ import annotations

import asyncio

import pytest

from deepx.middleware.hitl import HumanInTheLoopHooks


class TestHumanInTheLoopHooks:
    def test_init(self):
        hitl = HumanInTheLoopHooks(["web_search", "execute"])
        assert "web_search" in hitl._sensitive
        assert "execute" in hitl._sensitive

    def test_init_empty(self):
        hitl = HumanInTheLoopHooks([])
        assert not hitl._sensitive

    @pytest.mark.asyncio
    async def test_gate_tool_skips_non_sensitive(self):
        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=lambda a, t: False)
        assert await hitl.gate_tool("agent", "read_file") is None

    @pytest.mark.asyncio
    async def test_gate_tool_approves_and_remembers(self):
        calls = 0

        def approve_once(agent_name, tool_name):
            nonlocal calls
            calls += 1
            return True

        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=approve_once)
        assert await hitl.gate_tool("a1", "web_search") is None
        assert await hitl.gate_tool("a2", "web_search") is None
        assert calls == 1
        assert "web_search" in hitl._approved

    @pytest.mark.asyncio
    async def test_gate_tool_rejection_returns_message(self):
        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=lambda a, t: False)
        msg = await hitl.gate_tool("agent", "web_search")
        assert msg is not None
        assert "declined" in msg.lower() or "Human-in-the-loop" in msg
        assert "web_search" in msg

    @pytest.mark.asyncio
    async def test_parallel_gate_only_prompts_once(self):
        call_count = 0

        def counting_approval(agent_name, tool_name):
            nonlocal call_count
            call_count += 1
            return True

        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=counting_approval)

        async def one():
            return await hitl.gate_tool("agent", "web_search")

        await asyncio.gather(one(), one(), one(), one())
        assert call_count == 1
        assert "web_search" in hitl._approved
