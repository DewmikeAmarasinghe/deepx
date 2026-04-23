"""Multi-line user input: first line exactly ``\"\"\"`` opens a block closed by ``\"\"\"``."""

from __future__ import annotations

_TRIPLE = '"""'


def read_user_turn() -> str:
    """Read one message.

    If the first line is exactly ``"\"\"`` (no backslashes), read lines until another line that is only
    ``\"\"\"`` (after strip). A line consisting of only ``\"\"\"`` inside the body ends the block early.
    """
    first = input()
    if first.strip() == _TRIPLE:
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                return "\n".join(lines).strip()
            if line.strip() == _TRIPLE:
                return "\n".join(lines).strip()
            lines.append(line)
    return first.strip()


__all__ = ["read_user_turn"]
