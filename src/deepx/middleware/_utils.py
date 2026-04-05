from __future__ import annotations


LARGE_OUTPUT_THRESHOLD = 80_000
PREVIEW_LINES = 30


def generate_preview(content: str, max_lines: int = PREVIEW_LINES) -> str:
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    preview_lines = lines[:max_lines]
    omitted = len(lines) - max_lines
    return "\n".join(preview_lines) + f"\n\n... [{omitted} more lines omitted]"


def sanitize_path_component(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64]
