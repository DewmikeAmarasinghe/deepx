"""Password auth for the demo Chainlit app."""

from __future__ import annotations

import chainlit as cl


@cl.password_auth_callback
async def password_auth_callback(username: str, password: str) -> cl.User | None:
    if username == "admin" and password == "admin":
        return cl.User(
            identifier="admin",
            display_name="Admin",
            metadata={"role": "admin"},
        )
    return None
