"""Multi-agent orchestrator demo.

Run from the **repository root** so paths like ``./test_demo/skills/...`` resolve correctly.

Setup::
    uv sync --extra demo

    python -m test_demo.orchestrator                    # streaming chat (default)
    python -m test_demo.orchestrator --chat             # same: stream assistant tokens
    python -m test_demo.orchestrator --chat_sync        # chat without token streaming
    python -m test_demo.orchestrator --chat --session <id>     # resume (``--chat_sync`` also ok)

**Resume:** pass **``--session <id>``** together with **``--chat``** or **``--chat_sync``**.

In the REPL: **Enter** sends the message; **/bye** exits and prints how to resume with the same id.

Installing the ``deepx`` distribution exposes ``deepx`` and ``deepx_cli``; this orchestrator
lives under ``test_demo`` for deepx demonstration purposes only.
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents import RunContextWrapper, function_tool  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.text import Text  # noqa: E402

from deepx import create_deep_agent  # noqa: E402
from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx.backends.protocol import BackendProtocol  # noqa: E402
from deepx.context import AgentContext  # noqa: E402
from test_demo.hf_agent import hf_agent_runner  # noqa: E402
from test_demo.pdf_agent import pdf_agent_runner  # noqa: E402
from test_demo.sql_agent import sql_agent_runner  # noqa: E402
from test_demo.web_agent import web_agent_runner  # noqa: E402

_PREVIEW_HEAD_LINES = 70
_PREVIEW_TAIL_LINES = 35
_MAX_PDF_PAGES_PREVIEW = 2
_render_console = Console(highlight=False)


def _line_preview_body(text: str, *, head: int, tail: int) -> str:
    lines = text.splitlines()
    n = len(lines)
    if n <= head + tail:
        return text
    mid_omitted = n - head - tail
    return (
        "\n".join(lines[:head])
        + f"\n\n… ({mid_omitted} lines omitted) …\n\n"
        + "\n".join(lines[-tail:])
    )


def _preview_plain_file(backend: BackendProtocol, sid: str, agent_path: str) -> str:
    rr = backend.read(sid, agent_path, 0, _PREVIEW_HEAD_LINES)
    if rr.error:
        return rr.error
    total = rr.total_lines
    head = rr.content or ""
    if total is None:
        return _line_preview_body(
            head, head=_PREVIEW_HEAD_LINES, tail=_PREVIEW_TAIL_LINES
        )
    if total <= _PREVIEW_HEAD_LINES:
        return head
    tail_start = max(_PREVIEW_HEAD_LINES, total - _PREVIEW_TAIL_LINES)
    rr2 = backend.read(sid, agent_path, tail_start, total - tail_start)
    tail = rr2.content or ""
    omitted = tail_start - _PREVIEW_HEAD_LINES
    if omitted > 0:
        return head + f"\n\n… ({omitted} lines omitted) …\n\n" + tail
    return head + "\n\n" + tail


def render_pdf_as_text(
    ctx: RunContextWrapper[AgentContext], sid: str, agent_path: str
) -> str:
    """Extract text from a PDF using pdfplumber (bounded for terminal display)."""
    try:
        import pdfplumber
    except ImportError:
        return "[pdfplumber not installed — cannot render PDF as text]"

    resolved = ctx.context.backend.resolve_path(sid, agent_path)
    if resolved is None or not Path(resolved).exists():
        return f"File not found: {agent_path}"

    try:
        pages: list[str] = []
        with pdfplumber.open(resolved) as pdf:
            n_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                if i > _MAX_PDF_PAGES_PREVIEW:
                    rest = n_pages - _MAX_PDF_PAGES_PREVIEW
                    pages.append(f"… ({rest} more pages omitted from preview) …")
                    break
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"── Page {i} ──\n{text}")

        if not pages:
            return "(PDF has no extractable text — may be scanned/image-only)"

        full = "\n\n".join(pages)
        return _line_preview_body(
            full, head=_PREVIEW_HEAD_LINES, tail=_PREVIEW_TAIL_LINES
        )

    except Exception as exc:
        return f"PDF render error: {exc}"


@function_tool
async def render_files(
    ctx: RunContextWrapper[AgentContext],
    paths: list[str],
) -> str:
    """Show finished file contents to the human in the terminal (Rich panels).

    The user sees the **full** rendered content for each path in their terminal. Call **once**
    when work is **complete** (or when they should see final outputs), with **every** relevant path.
    Do **not** use this for exploratory reading while delegating.

    After this returns, **do not** tell the user you "rendered" or "displayed" the file, **do not**
    repeat the file body or long excerpts in chat, and **do not** offer to paste or open the same
    paths unless the user asks — they already saw it in the panel.
    """
    sid = ctx.context.session_id
    parts: list[str] = []
    w = _render_console.size.width or 120

    for path in paths:
        p = (path or "").strip()
        if not p:
            continue

        mime, _ = mimetypes.guess_type(p)
        is_pdf = (mime == "application/pdf") or p.lower().endswith(".pdf")

        if is_pdf:
            panel_content = render_pdf_as_text(ctx, sid, p)
        else:
            panel_content = _preview_plain_file(ctx.context.backend, sid, p)

        _render_console.print(
            Panel(
                Text(panel_content),
                title=Text(f"render_files · {p}"),
                border_style="yellow",
                expand=True,
                width=w,
            )
        )
        parts.append(p)

    return "Rendered: " + "; ".join(parts) if parts else "No paths provided."


DBS_DIR = REPO_ROOT / "test_demo" / "dbs"
TEST_DBS = DBS_DIR / "test_dbs"
AGENT_DBS = DBS_DIR / "agent_dbs"
for d in (TEST_DBS, AGENT_DBS):
    d.mkdir(parents=True, exist_ok=True)

DEMO_BACKEND = FilesystemBackend(REPO_ROOT)

ORCH_DB = str(AGENT_DBS / "orchestrator.db")

orch_tools = [render_files]

DEMO_ORCHESTRATOR_SYS = """\
**Coordinate specialists when the task matches the routing table below.** For those tasks you do **not**
run web/Tavily, SQL, PDF, or HF work yourself — you delegate with a **single, self-contained brief**
(including output paths, `db_name` when needed, and reasonable defaults like “top 3 with ties” if the user
already implied them).

**General work (no matching specialist):** answer from context, use your file tools, or guide the user.
Examples: skill/marketplace setup, LobeHub install steps, chit-chat, or anything that is **not** purely
web research, SQL on demo DBs, PDF workflows, or HF Hub MCP. This demo orchestrator has **no** host `execute`
tool — you cannot run shell commands yourself; use `read_file` / planning tools and clear instructions, or
route web execution to **`web_agent`** when the user needs live fetching.

**Delegate first; almost never ask the user to “confirm” the task shape.** If the user already stated
constraints (e.g. “top three genres and top three artists”), **do not** re-ask or paraphrase into clarifying
questions — pick sound defaults, state them in the brief, and call the specialist. Ask the user **only** when
something **objectively required** is missing (e.g. which `*.db` when several apply and the user gave none).

**Do not overcomplicate the task**. Limit the response to the smallest useful action that directly addresses the user’s request. Avoid expanding into broader analysis, multiple steps, or extra context unless the user asks for it.

**Specialist outputs:** In every delegation brief, tell the specialist to **write deliverables to files**
under **`/_outputs/`** (exact paths you agree on, e.g. `/_outputs/diffusion_models_summary.md`) and to
return **only paths plus a short summary** — not full file bodies.

**After a specialist tool returns:** trust its **paths** and **short summary**. Do **not** `read_file` those
files. To return those outputs to the user, call **`render_files`** once with **all** final paths.

**After `render_files`:** the user has seen the **full** files in the terminal. Reply with a **compact**
shortlist, **paths**, and **recommendation** only — **no** “I showed you the file”, **no** repeating the
digest, **no** offering to paste or open the same paths unless the user explicitly asks.

**If a specialist’s reply contains a question for the user:** relay it briefly; do not stack your own
extra questions on top.

## Routing

| Goal | Use |
|------|-----|
| Live web research, citations, long-form markdown from the open web | **`web_agent` tool** — one self-contained brief; specialist uses `tvly`, tavily skills, and **write-report** for deliverables. |
| Read-only SQL on sample ``*.db`` files under `test_demo/dbs/test_dbs` | **`sql_agent` tool** — self-contained brief; every query needs **`db_name`** (e.g. `chinook.db`). |
| PDF workflows | **`pdf_agent` tool**. |
| Hugging Face Hub | **`hf_agent` tool** when configured (**HF_TOKEN**); keep delegation brief. |
| Show finished text to the user | **`render_files`** once, with every relevant path — not for mid-task browsing. |

**Delegation:** one strong **`tool` call per specialist** when possible; pass **paths** between
steps, not pasted bodies.
"""

_ORCH_SUBAGENTS = [
    web_agent_runner,
    sql_agent_runner,
    pdf_agent_runner,
]
if hf_agent_runner is not None:
    _ORCH_SUBAGENTS.append(hf_agent_runner)


orchestrator_runner = create_deep_agent(
    name="orchestrator",
    description=(
        "Coordinates **web_agent** (Tavily CLI + report standards), **sql_agent**, **pdf_agent**, and "
        "optional **hf_agent**. Delegates with self-contained briefs; specialists write under **/_outputs/**. "
        "You plan, route, and use **render_files**—you do not replace specialists for their domains."
    ),
    subagents=_ORCH_SUBAGENTS,
    memory=[".deepx/AGENTS.md"],
    tools=orch_tools,
    skills=[str(REPO_ROOT / "_outputs" / "weather_task_output" / "skills")],
    system_prompt=DEMO_ORCHESTRATOR_SYS,
    checkpointer=ORCH_DB,
    debug=True,
    backend=DEMO_BACKEND,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepx orchestrator demo (terminal).")
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Resume this session id (must be used with --chat or --chat_sync).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn session with streaming (default if --chat_sync not set).",
    )
    parser.add_argument(
        "--chat_sync",
        action="store_true",
        help="Interactive multi-turn session without token streaming.",
    )
    args, _rest = parser.parse_known_args()

    if args.session is not None and not args.chat and not args.chat_sync:
        parser.error("--session requires --chat or --chat_sync")

    from deepx_cli.chat_stream import run_chat_stream
    from deepx_cli.chat_sync import run_chat_sync

    if not args.chat_sync:
        run_chat_stream(orchestrator_runner)
    else:
        run_chat_sync(orchestrator_runner)


if __name__ == "__main__":
    main()
