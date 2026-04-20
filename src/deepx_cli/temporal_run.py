from __future__ import annotations


async def run_via_temporal(
    *,
    prompt: str,
    session_id: str,
    resume: bool,
) -> str:
    try:
        from test_demo.temporal.client import run_orchestrator_workflow_and_wait
    except ImportError as e:
        msg = (
            "USE_TEMPORAL is set but the demo package could not be imported. "
            "Run from the repository root with `uv run --extra demo python -m test_demo.orchestrator`."
        )
        raise RuntimeError(msg) from e
    return await run_orchestrator_workflow_and_wait(
        prompt,
        session_id,
        resume=resume,
    )


__all__ = ["run_via_temporal"]
