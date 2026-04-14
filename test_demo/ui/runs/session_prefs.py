"""Session-level defaults for the Chainlit demo (avoid import cycles)."""

from __future__ import annotations

import os


def default_use_temporal() -> bool:
    v = os.environ.get("DEEPX_USE_TEMPORAL", "true").strip().lower()
    return v in ("1", "true", "yes", "on")
