import asyncio
import subprocess

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext


@function_tool
async def execute_command(
    ctx: RunContextWrapper[AgentContext], command: str, timeout: int = 30
) -> str:

    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    p = await asyncio.to_thread(_run)
    parts = []
    if p.stdout:
        parts.append(p.stdout)
    if p.stderr:
        for line in p.stderr.splitlines():
            parts.append(f"[stderr] {line}")
    out = "\n".join(parts) if parts else ""
    if p.returncode != 0:
        out = out + f"\n[exit code: {p.returncode}]" if out else f"[exit code: {p.returncode}]"
    return out
