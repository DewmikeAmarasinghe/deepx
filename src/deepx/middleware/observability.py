from __future__ import annotations

import os


def setup_observability() -> None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "deepx")
    try:
        from langsmith.integrations.openai_agents_sdk import (
            OpenAIAgentsTracingProcessor,
        )
        from agents.tracing import set_trace_processors

        set_trace_processors([OpenAIAgentsTracingProcessor()])  # type: ignore[list-item]
    except Exception:
        return
