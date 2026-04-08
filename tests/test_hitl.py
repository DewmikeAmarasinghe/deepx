"""Unit tests for HITL approval persistence."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepx.backends.memory import InMemoryBackend
from deepx.context import AgentContext
from deepx.middleware.hitl import HumanInTheLoopHooks


def _make_ctx(agent_name: str = "agent") -> MagicMock:
    backend = InMemoryBackend()
    agent_ctx = AgentContext(
        session_id="s1",
        backend=backend,
        agent_name=agent_name,
    )
    ctx = MagicMock()
    ctx.context = agent_ctx
    return ctx


def _make_agent(name: str = "agent") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


class TestHumanInTheLoopHooks:
    def test_init(self):
        hitl = HumanInTheLoopHooks(["web_search", "execute"])
        assert "web_search" in hitl._sensitive
        assert "execute" in hitl._sensitive

    def test_init_empty(self):
        hitl = HumanInTheLoopHooks([])
        assert not hitl._sensitive

    def test_approved_stored_on_hitl_instance(self):
        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=lambda a, t: True)
        ctx = _make_ctx()
        agent = _make_agent()
        tool = _make_tool("web_search")
        asyncio.run(hitl.on_tool_start(ctx, agent, tool))
        assert "web_search" in hitl._approved

    def test_non_sensitive_tool_skipped(self):
        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=lambda a, t: True)
        ctx = _make_ctx()
        agent = _make_agent()
        tool = _make_tool("read_file")
        asyncio.run(hitl.on_tool_start(ctx, agent, tool))
        assert "read_file" not in hitl._approved

    def test_second_call_skips_prompt(self):
        call_count = 0

        def counting_approval(agent_name, tool_name):
            nonlocal call_count
            call_count += 1
            return True

        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=counting_approval)
        ctx = _make_ctx()
        agent = _make_agent()
        tool = _make_tool("web_search")

        asyncio.run(hitl.on_tool_start(ctx, agent, tool))
        asyncio.run(hitl.on_tool_start(ctx, agent, tool))
        asyncio.run(hitl.on_tool_start(ctx, agent, tool))

        assert call_count == 1

    def test_approval_shared_across_different_contexts(self):
        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=lambda a, t: True)
        ctx1 = _make_ctx("agent1")
        ctx2 = _make_ctx("agent2")
        agent = _make_agent()
        tool = _make_tool("web_search")

        asyncio.run(hitl.on_tool_start(ctx1, agent, tool))
        assert "web_search" in hitl._approved

        call_count = 0

        def counting_approval(agent_name, tool_name):
            nonlocal call_count
            call_count += 1
            return True

        hitl._approval_fn = counting_approval
        asyncio.run(hitl.on_tool_start(ctx2, agent, tool))
        assert call_count == 0

    def test_rejection_raises(self):
        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=lambda a, t: False)
        ctx = _make_ctx()
        agent = _make_agent()
        tool = _make_tool("web_search")
        with pytest.raises(RuntimeError, match="rejected"):
            asyncio.run(hitl.on_tool_start(ctx, agent, tool))

    def test_parallel_calls_only_prompt_once(self):
        call_count = 0

        def counting_approval(agent_name, tool_name):
            nonlocal call_count
            call_count += 1
            return True

        hitl = HumanInTheLoopHooks(["web_search"], approval_fn=counting_approval)
        ctx = _make_ctx()
        agent = _make_agent()
        tool = _make_tool("web_search")

        async def run_parallel():
            await asyncio.gather(
                hitl.on_tool_start(ctx, agent, tool),
                hitl.on_tool_start(ctx, agent, tool),
                hitl.on_tool_start(ctx, agent, tool),
                hitl.on_tool_start(ctx, agent, tool),
            )

        asyncio.run(run_parallel())
        assert call_count == 1
        assert "web_search" in hitl._approved
