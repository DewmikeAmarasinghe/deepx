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
        "PDF and form workflows: reading, merging, splitting, extracting tables/text, fillable "
        "forms; may use `execute` for bundled scripts when appropriate."
    ),
    tools=[],
    skills=[str(PDF_SKILLS_DIR)],
    system_prompt=(
        "You specialise in PDF tasks. For multi-step work, call **`write_todos` first** (after "
        "`read_file` on **pdf** / **forms** skill entry paths), then **`write_todos`** again with "
        "an updated full list as you go.\n"
        "Follow the skill workflows; use `execute` for repo scripts when the environment has the "
        "needed Python packages. Write outputs under sensible project paths and return paths.\n"
        "**OCR / scanned PDFs:** if the user needs OCR, say clearly that **Tesseract** (or similar) "
        "must be installed on the host system—pip packages alone are not enough."
    ),
    backend=_DEMO_BACKEND,
    checkpointer=_PDF_DB,
    debug=True,
    subagents=None,
)
