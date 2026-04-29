"""PDF / forms subagent (demo)."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.factory import create_deep_agent  # noqa: E402

REPO_ROOT = _REPO_ROOT

_DEMO_BACKEND = LocalShellBackend(REPO_ROOT)
_AGENT_DBS = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_PDF_DB = str(_AGENT_DBS / "pdf_agent.db")


pdf_agent_runner = create_deep_agent(
    name="pdf_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "PDF related tasks specialist"
        "extract text/tables, metadata, and forms workflows per **pdf** skills under "
        "`test_demo/skills/pdf`. Returns paths plus a short summary."
        "For scanned PDFs, OCR needs host tools (e.g. Tesseract)—state limits plainly."
    ),
    tools=[],
    skills=["./test_demo/skills/pdf"],
    system_prompt=(
        "You specialise in PDF and forms. **`read_file`** skill `SKILL.md` entries first; use **`execute`** to run "
        "companion scripts under the skill tree (e.g. `python /test_demo/skills/pdf/.../script.py ...`) when documented.\n\n"
        "Use **file tools** for reading/writing project files.\n\n"
        "For multi-step work: **`write_todos`** after skilling up; refresh the list as steps complete.\n\n"
        "**OCR / scanned PDFs:** if the user needs OCR, say clearly that **Tesseract** (or similar) must be on the host."
    ),
    backend=_DEMO_BACKEND,
    checkpointer=_PDF_DB,
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)


def main() -> None:
    from deepx_cli.cli import run_interactive_cli

    run_interactive_cli(
        pdf_agent_runner,
        description="Deepx PDF specialist demo (terminal).",
    )


if __name__ == "__main__":
    main()
