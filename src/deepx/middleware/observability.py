from __future__ import annotations

import os
from typing import cast


def setup_observability() -> None:
    """Register LangSmith tracing for the OpenAI Agents SDK when credentials exist."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "deepx")
    try:
        from agents.tracing import TracingProcessor, set_trace_processors
        from langsmith.integrations.openai_agents_sdk import (
            OpenAIAgentsTracingProcessor,
        )

        set_trace_processors(
            cast(list[TracingProcessor], [OpenAIAgentsTracingProcessor()])
        )
    except Exception:
        return
