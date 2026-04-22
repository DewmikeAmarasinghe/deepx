from __future__ import annotations

import os


def setup_observability() -> None:
    """Register LangSmith tracing for the OpenAI Agents SDK when credentials exist.

    Does not enable tracing by itself: respects ``LANGSMITH_TRACING`` / ``LANGCHAIN_TRACING_V2``
    if already set; otherwise only registers the processor when a LangSmith API key is present.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return
    # Do not override explicit user choice
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in (
        "false",
        "0",
        "no",
        "off",
    ):
        return
    if os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in (
        "false",
        "0",
        "no",
        "off",
    ):
        return
    if not os.getenv("LANGSMITH_TRACING", "").strip():
        os.environ["LANGSMITH_TRACING"] = "true"
    try:
        from langsmith.integrations.openai_agents_sdk import (
            OpenAIAgentsTracingProcessor,
        )
        from agents.tracing import set_trace_processors

        set_trace_processors([OpenAIAgentsTracingProcessor()])  # type: ignore[list-item]
    except Exception:
        return
