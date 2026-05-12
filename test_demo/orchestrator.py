"""Multi-agent orchestrator demo.

Run from the **repository root** so paths like ``./test_demo/skills/...`` resolve correctly.

Setup::

    uv sync --extra demo

    python -m test_demo.orchestrator --chat
    python -m test_demo.orchestrator --chat_sync
    python -m test_demo.orchestrator --chat --session <id>
"""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents import RunContextWrapper, function_tool  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.text import Text  # noqa: E402

from deepx import create_deep_agent  # noqa: E402
from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.context import AgentContext  # noqa: E402
from test_demo.hf_agent import hf_agent_runner  # noqa: E402
from test_demo.pdf_agent import pdf_agent_runner  # noqa: E402
from test_demo.sql_agent import sql_agent_runner  # noqa: E402
from test_demo.web_agent import web_agent_runner  # noqa: E402

_PREVIEW_HEAD = 35
_PREVIEW_TAIL = 35
_MAX_PDF_PAGES = 2
_render_console = Console(highlight=False)

_AGENT_DBS = _REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_TEST_DBS = _REPO_ROOT / "test_demo" / "dbs" / "test_dbs"


# ---------------------------------------------------------------------------
# File preview helpers
# ---------------------------------------------------------------------------


def _truncated_preview(text: str) -> str:
    lines = text.splitlines()
    n = len(lines)
    if n <= _PREVIEW_HEAD + _PREVIEW_TAIL:
        return text
    omitted = n - _PREVIEW_HEAD - _PREVIEW_TAIL
    return (
        "\n".join(lines[:_PREVIEW_HEAD])
        + f"\n\n… ({omitted} lines omitted) …\n\n"
        + "\n".join(lines[-_PREVIEW_TAIL:])
    )


def _read_plain_file(backend: LocalShellBackend, sid: str, path: str) -> str:
    rr = backend.read(sid, path, 0, _PREVIEW_HEAD)
    if rr.error:
        return rr.error
    total = rr.total_lines
    head = rr.content or ""
    if total is None:
        return _truncated_preview(head)
    if total <= _PREVIEW_HEAD:
        return head
    tail_start = max(_PREVIEW_HEAD, total - _PREVIEW_TAIL)
    rr2 = backend.read(sid, path, tail_start, total - tail_start)
    tail = rr2.content or ""
    omitted = tail_start - _PREVIEW_HEAD
    if omitted > 0:
        return head + f"\n\n… ({omitted} lines omitted) …\n\n" + tail
    return head + "\n\n" + tail


def _read_pdf_file(ctx: RunContextWrapper[AgentContext], sid: str, path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        return "[pdfplumber not installed — cannot render PDF as text]"

    resolved = ctx.context.backend.resolve_path(sid, path)
    if resolved is None or not Path(resolved).exists():
        return f"File not found: {path}"

    try:
        pages: list[str] = []
        with pdfplumber.open(resolved) as pdf:
            n_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                if i > _MAX_PDF_PAGES:
                    pages.append(f"… ({n_pages - _MAX_PDF_PAGES} more pages omitted) …")
                    break
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"── Page {i} ──\n{text}")
        if not pages:
            return "(PDF has no extractable text — may be scanned/image-only)"
        return _truncated_preview("\n\n".join(pages))
    except Exception as exc:
        return f"PDF render error: {exc}"


# ---------------------------------------------------------------------------
# render_files tool
# ---------------------------------------------------------------------------


@function_tool
async def render_files(
    ctx: RunContextWrapper[AgentContext],
    paths: list[str],
) -> str:
    """Show finished file contents to the human in the terminal (Rich panels).

    The user sees the full rendered content for each path. Call once when work is complete,
    passing every relevant path. Do not use this for mid-task browsing.

    After this returns: do not repeat the file body, do not say you "rendered" it, and do not
    offer to open the same paths again unless the user asks.
    """
    sid = ctx.context.session_id
    backend = ctx.context.backend
    width = _render_console.size.width or 120
    rendered: list[str] = []

    for raw_path in paths:
        path = (raw_path or "").strip()
        if not path:
            continue

        mime, _ = mimetypes.guess_type(path)
        is_pdf = (mime == "application/pdf") or path.lower().endswith(".pdf")

        if is_pdf:
            content = _read_pdf_file(ctx, sid, path)
        else:
            if isinstance(backend, LocalShellBackend):
                content = _read_plain_file(backend, sid, path)
            else:
                content = f"[Unsupported backend type for plain file reading: {type(backend).__name__}]"

        _render_console.print(
            Panel(
                Text(content),
                title=Text(f"render_files · {path}"),
                border_style="yellow",
                expand=True,
                width=width,
            )
        )
        rendered.append(path)

    return "Rendered: " + "; ".join(rendered) if rendered else "No paths provided."


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

orch_sys = """\
You are a personal assistant.

When a task requires web research, SQL, PDF work, or Hugging Face Hub access, coordinate
the appropriate specialists below.

**Act, don't ask.** If the user's request gives you enough to act on, pick sensible
defaults and start — do not ask for confirmation. Only ask when something objectively
required is missing (e.g. which database when several apply and the user gave no hint).

**Keep scope tight.** Do the smallest useful thing that directly addresses the request.
Do not expand into broader analysis unless the user asks for it.

**File-based pipelines — chain specialists through paths, not pasted content.**
When one specialist's output feeds the next, pass the file paths in the second brief.
Example: user asks for research compiled into a PDF →
  1. Brief web_agent: research the topic, write a markdown report to `/_outputs/report.md`.
  2. Brief pdf_agent: read `/_outputs/report.md` and produce `/_outputs/report.pdf` from it.
  3. Call render_files with `/_outputs/report.pdf`.
Apply this pattern for any multi-specialist workflow: SQL → report, research → slides, etc.

**After a specialist returns:** trust its paths and short summary. Do not read those files
yourself unless you need them for a downstream task. When work is complete, call render_files
once with all final output paths.

**After render_files:** the user has already seen the files. Reply with a compact summary,
the paths, and any recommendation — nothing more.

**If a specialist replies with a question for the user:** relay it briefly; do not add
your own questions on top.

## Specialist routing

| Goal | Specialist |
|------|-----------|
| Live web research, news, documentation, long-form markdown reports | web_agent |
| SQL queries, schema design, migrations, database management | sql_agent |
| PDF creation, extraction, merging, form filling | pdf_agent |
| Hugging Face Hub: model/dataset/space search, Hub docs | hf_agent (when HF_TOKEN is set) |
| Skill authoring or SKILL.md structure | skill-creator skill (handle directly) |
| Cron expressions, scheduling, crontab snippets | cron-scheduling skill (handle directly) |
| Show finished output to the user | render_files — once, with all final paths |
"""

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

_TEST_DBS.mkdir(parents=True, exist_ok=True)
_AGENT_DBS.mkdir(parents=True, exist_ok=True)

subagents = [web_agent_runner, sql_agent_runner, pdf_agent_runner]
if hf_agent_runner is not None:
    subagents.append(hf_agent_runner)

orchestrator_runner = create_deep_agent(
    name="orchestrator",
    memory=[".deepx/AGENTS.md"],
    description=(
        "Personal assistant that coordinates web_agent (live research), sql_agent (SQLite "
        "queries), pdf_agent (PDF creation and extraction), and optionally hf_agent (Hugging "
        "Face Hub). Delegates with self-contained briefs, chains specialist outputs through "
        "file paths, and surfaces final results via render_files."
    ),
    subagents=subagents,
    tools=[render_files],
    skills=[
        "./test_demo/skills/skill-creator",
        "./test_demo/skills/cron-scheduling",
    ],
    system_prompt=orch_sys,
    backend=LocalShellBackend(_REPO_ROOT),
    checkpointer=str(_AGENT_DBS / "orchestrator.db"),
    debug=True,
    interrupt_on=["execute"],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from deepx_cli import run_interactive_cli
    run_interactive_cli(orchestrator_runner, description="Deepx orchestrator demo.")


if __name__ == "__main__":
    main()