"""Chainlit persistence: ``SQLAlchemyDataLayer`` plus idempotent SQLite schema."""

from __future__ import annotations

import os
import warnings

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

from test_demo.ui.bootstrap.paths import REPO_ROOT
from test_demo.ui.persistence.chainlit_schema import ensure_chainlit_sqlite_schema


def chainlit_database_url() -> str:
    ui_dir = REPO_ROOT / "test_demo" / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    return os.environ.get(
        "CHAINLIT_DATABASE_URL",
        "sqlite+aiosqlite:///" + str((ui_dir / "chainlit.db").resolve()),
    )


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    url = chainlit_database_url()
    ensure_chainlit_sqlite_schema(url)
    secret = os.environ.get("CHAINLIT_AUTH_SECRET", "")
    if secret and len(secret) < 32:
        warnings.warn(
            "CHAINLIT_AUTH_SECRET should be at least 32 characters "
            "(e.g. `openssl rand -hex 32`) for stable HMAC signing.",
            stacklevel=1,
        )
    return SQLAlchemyDataLayer(conninfo=url)
