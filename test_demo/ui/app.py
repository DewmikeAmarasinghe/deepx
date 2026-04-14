"""Chainlit entry: ensure repo is importable, load env, register callbacks."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv()

from test_demo.ui.auth import password_auth  # noqa: F401
from test_demo.ui.chat import profiles  # noqa: F401
from test_demo.ui.chat import session_hooks  # noqa: F401
from test_demo.ui.chat import settings_widgets  # noqa: F401
from test_demo.ui.persistence import data_layer  # noqa: F401
from test_demo.ui.runs import run_modes  # noqa: F401
