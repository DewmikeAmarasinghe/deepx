"""Deepx middleware.

Lazy :func:`__getattr__` keeps :mod:`deepx.context` able to import from
:mod:`deepx.middleware.hitl` without loading heavy middleware submodules at package import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepx.middleware.filesystem import FilesystemHooks
    from deepx.middleware.logs import SessionToolLogHooks
    from deepx.middleware.tool_pipeline import (
        apply_tool_pipeline,
        wrap_tools_for_large_tool_results,
    )
    from deepx.middleware.hitl import (
        DEFAULT_REJECTION_MESSAGE,
        Hitl,
        HitlCallback,
        HitlDecision,
        HitlRequest,
        wrap_tools_for_hitl,
    )
    from deepx.middleware.observability import setup_observability
    from deepx.middleware.run_hooks import ChainedRunHooks, compose_run_hooks

__all__ = [
    "ChainedRunHooks",
    "compose_run_hooks",
    "FilesystemHooks",
    "Hitl",
    "HitlCallback",
    "HitlDecision",
    "HitlRequest",
    "DEFAULT_REJECTION_MESSAGE",
    "SessionToolLogHooks",
    "apply_tool_pipeline",
    "setup_observability",
    "wrap_tools_for_hitl",
    "wrap_tools_for_large_tool_results",
]


def __getattr__(name: str) -> Any:
    if name == "FilesystemHooks":
        from deepx.middleware.filesystem import FilesystemHooks

        return FilesystemHooks
    if name in ("apply_tool_pipeline", "wrap_tools_for_large_tool_results"):
        import deepx.middleware.tool_pipeline as _tp

        return getattr(_tp, name)
    if name == "SessionToolLogHooks":
        from deepx.middleware.logs import SessionToolLogHooks

        return SessionToolLogHooks
    if name in (
        "Hitl",
        "HitlCallback",
        "HitlDecision",
        "HitlRequest",
        "DEFAULT_REJECTION_MESSAGE",
        "wrap_tools_for_hitl",
    ):
        import deepx.middleware.hitl as _hitl

        return getattr(_hitl, name)
    if name == "setup_observability":
        from deepx.middleware.observability import setup_observability

        return setup_observability
    if name in ("ChainedRunHooks", "compose_run_hooks"):
        from deepx.middleware import run_hooks as _rh

        return getattr(_rh, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
