from __future__ import annotations

import os


def setup_observability() -> None:
    """Configure Langfuse tracing via the OpenInference instrumentation layer.

    Reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and optionally
    ``LANGFUSE_BASE_URL`` from the environment.  Call this **after**
    ``load_dotenv()`` so the variables are already set.

    Uses the Langfuse SDK's own tracer provider rather than raw OTLP env vars,
    which is more reliable and avoids header-format issues.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print(
            "[deepx] Langfuse not configured — "
            "set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable tracing."
        )
        return

    try:
        from langfuse import Langfuse
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

        lf = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        if lf._resources and lf._resources.tracer_provider:
            OpenAIAgentsInstrumentor().instrument(
                tracer_provider=lf._resources.tracer_provider
            )
            print(f"[deepx] Langfuse tracing enabled → {host}")
        else:
            print("[deepx] Langfuse connected but tracer provider unavailable.")
    except ImportError as e:
        print(f"[deepx] Langfuse tracing skipped — missing package: {e}")