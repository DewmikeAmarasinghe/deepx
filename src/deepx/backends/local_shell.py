"""Host filesystem + unrestricted local shell (same spirit as LangChain deepagents LocalShellBackend)."""

from __future__ import annotations

import subprocess

from deepx.backends.filesystem import FilesystemBackend


class LocalShellBackend(FilesystemBackend):
    """FilesystemBackend with ``execute``: ``cwd`` is the host ``root_dir`` (not session workspace)."""

    def execute(
        self,
        session_id: str,
        command: str,
        *,
        timeout: float = 120.0,
        max_chars: int = 50_000,
    ) -> str:
        _ = session_id
        cmd = (command or "").strip()
        if not cmd:
            return "No command provided."
        cwd = str(self._host_root)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout} seconds."
        except OSError as e:
            return f"Error: {e}"
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > max_chars:
            out = out[:max_chars] + "\n...[truncated]"
        return f"exit_code={proc.returncode}\n{out}"
