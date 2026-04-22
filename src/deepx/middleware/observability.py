from __future__ import annotations

import os


def setup_observability() -> None:
    """Register LangSmith tracing for the OpenAI Agents SDK when credentials exist."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "default")
    try:
        from langsmith.integrations.openai_agents_sdk import (
            OpenAIAgentsTracingProcessor,
        )
        from agents.tracing import set_trace_processors

        set_trace_processors([OpenAIAgentsTracingProcessor()])  # type: ignore[list-item]
    except Exception:
        return
