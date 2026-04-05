from __future__ import annotations
import base64
import os


def setup_observability() -> None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("[deepx] Langfuse not configured — set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable tracing.")
        return

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{base_url}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
        OpenAIAgentsInstrumentor().instrument()
        print(f"[deepx] Langfuse tracing enabled → {base_url}")
    except ImportError:
        print("[deepx] openinference-instrumentation-openai-agents not installed — tracing disabled.")