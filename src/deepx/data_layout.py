"""Canonical `.deepx` data directory layout (shared by filesystem backend and run logs)."""

from __future__ import annotations

from pathlib import Path


def data_root_for_host(host_root: Path) -> Path:
    r = host_root.expanduser().resolve()
    if r.name == ".deepx":
        return r
    return r / ".deepx"
