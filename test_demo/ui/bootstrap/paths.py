"""Ensure the repository root is on ``sys.path`` and expose ``REPO_ROOT``."""

from __future__ import annotations

import sys
from pathlib import Path

# test_demo/ui/bootstrap/paths.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
