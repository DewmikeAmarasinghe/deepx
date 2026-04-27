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
    ) -> str:
        _ = session_id
        cmd = (command or "").strip()
        if not cmd:
            return "No command provided."
        cwd = str(self._root_dir)
        cap = min(float(timeout), 600.0)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=cap,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {cap} seconds."
        except OSError as e:
            return f"Error: {e}"
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"exit_code={proc.returncode}\n{out}"
