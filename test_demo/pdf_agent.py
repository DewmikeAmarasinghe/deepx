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

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _REPO_ROOT
PDF_SKILLS_DIR = DEMO_DIR / "skills" / "pdf"

_DEMO_BACKEND = LocalShellBackend(REPO_ROOT)
_AGENT_DBS = REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_PDF_DB = str(_AGENT_DBS / "pdf_agent.db")


pdf_agent_runner = create_deep_agent(
    name="pdf_agent",
    description=(
        "PDF related tasks specialist"
        "extract text/tables, metadata, and forms workflows per **pdf** skills under "
        "`test_demo/skills/pdf`. Writes outputs to agreed paths (typically **/_outputs/**). "
        "For scanned PDFs, OCR needs host tools (e.g. Tesseract)—state limits plainly."
    ),
    tools=[],
    skills=[str(PDF_SKILLS_DIR)],
    system_prompt=(
        "You specialise in PDF tasks. For multi-step work, call **`write_todos` first** (after "
        "`read_file` on **pdf** / **forms** skill entry paths), then **`write_todos`** again with "
        "an updated full list as you go.\n"
        "Follow the skill workflows; use file tools and skills (no host shell in this agent). "
        "Write outputs under sensible project paths and return paths.\n"
        "**OCR / scanned PDFs:** if the user needs OCR, say clearly that **Tesseract** (or similar) "
        "must be installed on the host system—pip packages alone are not enough."
    ),
    backend=_DEMO_BACKEND,
    checkpointer=_PDF_DB,
    debug=True,
    subagents=None,
)
