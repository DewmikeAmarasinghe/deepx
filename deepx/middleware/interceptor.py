import dataclasses
import functools
import json
from datetime import UTC, datetime
from typing import Any

from agents import FunctionTool
from agents.tool_context import ToolContext


class ToolInterceptor:
    EVICTION_THRESHOLD: int = 40_000

    @classmethod
    def wrap(cls, fn, tool_name: str):
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any):
            result = await fn(*args, **kwargs)
            ctx = args[0]
            return cls._finalize(ctx, tool_name, "", result)

        return wrapper

    @classmethod
    def _finalize(cls, ctx: Any, tool_name: str, input_summary: str, result: Any) -> Any:
        ac = ctx.context
        s = str(result)
        ac._step_counter += 1
        step = ac._step_counter
        obs_path = f"/obs/{step:04d}_{tool_name}.json"
        full_path = ""
        out_path = ""
        if len(s) > cls.EVICTION_THRESHOLD:
            out_path = f"/outputs/{step:04d}_{tool_name}.md"
            ac.vfs[out_path] = s
            full_path = out_path
        preview = s[:300]
        obs_payload = {
            "tool_name": tool_name,
            "step": step,
            "input_summary": input_summary,
            "result_preview": preview,
            "full_path": full_path,
        }
        ac.vfs[obs_path] = json.dumps(obs_payload, indent=2)
        ac.step_log.append(
            {
                "step": step,
                "tool": tool_name,
                "input_summary": input_summary,
                "preview": preview,
                "full_path": full_path,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        if len(s) > cls.EVICTION_THRESHOLD:
            lines = s.splitlines()
            preview_lines = "\n".join(lines[:8])
            return (
                f"Result saved to {out_path} ({len(lines)} lines).\n"
                f"Preview:\n{preview_lines}\n"
                f"...\nUse read_file to access the full result."
            )
        return result

    @classmethod
    def apply(cls, tools: list, ctx_ref: Any = None) -> list:
        out = []
        for tool in tools:
            if not isinstance(tool, FunctionTool):
                out.append(tool)
                continue
            orig = tool.on_invoke_tool
            name = tool.name

            async def wrapped(
                tc: ToolContext,
                input_json: str,
                _orig=orig,
                _name=name,
            ):
                result = await _orig(tc, input_json)
                summary = input_json[:2000] if input_json else ""
                return cls._finalize(tc, _name, summary, result)

            out.append(dataclasses.replace(tool, on_invoke_tool=wrapped))
        return out
