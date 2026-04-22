from __future__ import annotations

import asyncio

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext

EXECUTE_TIMEOUT_CAP_S = 600.0
EXECUTE_MAX_PARALLEL = 5


def _format_parallel_results(pairs: list[tuple[str, str]]) -> str:
    """Human- and model-friendly layout: one clearly delimited block per command."""
    n = len(pairs)
    lines: list[str] = [
        "=" * 72,
        f"execute — {n} shell command(s) ran in parallel (blocks below are in the same order as `commands`)",
        "=" * 72,
    ]
    for i, (cmd, block) in enumerate(pairs, start=1):
        lines.extend(
            [
                "",
                f"### [{i}/{n}] Command",
                "```",
                cmd,
                "```",
                f"### [{i}/{n}] Output (stdout+stderr, exit_code line first if present)",
                "```",
                block.rstrip(),
                "```",
            ]
        )
    lines.extend(["", "=" * 72, "End of execute output", "=" * 72])
    return "\n".join(lines)


@function_tool
async def execute(
    ctx: RunContextWrapper[AgentContext],
    commands: list[str],
    timeout_seconds: float = 120.0,
) -> str:
    """Run up to **5** host shell commands **in parallel** (same working directory: project root).

    Pass **`commands`**: a list of non-empty shell strings (e.g. multiple `tvly ...` calls at once).
    Each command uses the same **`timeout_seconds`** (clamped). Unsupported backends return a clear
    error. Very large combined output may be spilled to **`/_outputs/large_tool_results/`** by the
    framework with a preview in the tool message — must use **`read_file`** on the given path if so.
    """
    b = ctx.context.backend
    sid = ctx.context.session_id
    raw = [str(c).strip() for c in (commands or []) if str(c).strip()]
    if not raw:
        return "No commands provided. Pass a non-empty `commands` list."
    if len(raw) > EXECUTE_MAX_PARALLEL:
        return (
            f"Too many commands ({len(raw)}); maximum is {EXECUTE_MAX_PARALLEL} per `execute` call. "
            "Split into multiple execute calls."
        )
    t = float(timeout_seconds)
    t = min(max(t, 1.0), EXECUTE_TIMEOUT_CAP_S)

    def run_one(cmd: str) -> tuple[str, str]:
        return cmd, b.execute(sid, cmd, timeout=t)

    pairs: list[tuple[str, str]] = list(
        await asyncio.gather(*[asyncio.to_thread(run_one, c) for c in raw])
    )
    return _format_parallel_results(pairs)
