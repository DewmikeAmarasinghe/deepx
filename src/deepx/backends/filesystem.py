from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import wcmatch.glob as wcglob
from wcmatch import fnmatch as wc_fnmatch

from deepx.backends.protocol import (
    OUTPUTS_PREFIX,
    BackendProtocol,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepx.backends.utils import (
    _coerce_agent_path,
    _norm_agent_path,
    _physical_for_tools,
    _under_data_root,
    data_root_for_host,
)

MAX_GREP_FILE_BYTES = 10 * 1024 * 1024


def _flags_nofollow(base: int) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        return base | os.O_NOFOLLOW
    return base


def _read_file_bytes(path: Path) -> bytes:
    fd = os.open(str(path), _flags_nofollow(os.O_RDONLY))
    with os.fdopen(fd, "rb") as f:
        return f.read()


def _write_file_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        str(path), _flags_nofollow(os.O_CREAT | os.O_TRUNC | os.O_WRONLY), 0o644
    )
    with os.fdopen(fd, "wb") as f:
        f.write(data)


class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._data_root = data_root_for_host(self._root_dir)

    @property
    def data_root(self) -> Path:
        return self._data_root

    def _canon(self, agent_path: str) -> str:
        return _norm_agent_path(_coerce_agent_path(self._root_dir, agent_path))

    @staticmethod
    def _is_deepx_agent_path(canon: str) -> bool:
        return canon == "/.deepx" or canon.startswith("/.deepx/")

    def _physical(self, agent_path: str) -> tuple[Path | None, str | None]:
        canon = self._canon(agent_path)
        if self._is_deepx_agent_path(canon):
            suffix = canon[len("/.deepx") :].lstrip("/")
            try:
                if suffix:
                    p = (self._data_root / suffix).resolve()
                else:
                    p = self._data_root.resolve()
                p.relative_to(self._data_root.resolve())
            except (ValueError, OSError):
                return None, "Error: invalid .deepx path."
            return p, None
        return _physical_for_tools(self._root_dir, self._data_root, agent_path)

    def _agent_path_for_file(self, f: Path) -> str | None:
        try:
            rel = f.resolve().relative_to(self._root_dir.resolve())
            return "/" + rel.as_posix()
        except ValueError:
            return None

    def _agent_path_for_deepx_file(self, f: Path) -> str | None:
        try:
            rel = f.resolve().relative_to(self._data_root.resolve())
        except ValueError:
            return None
        return "/.deepx/" + rel.as_posix()

    def resolve_path(self, session_id: str, agent_path: str) -> str | None:
        _ = session_id
        p, err = self._physical(agent_path)
        if err or p is None:
            return None
        return str(p)

    def ls(self, session_id: str, path: str) -> LsResult:
        _ = session_id
        canon = self._canon(path)
        under_deepx = self._is_deepx_agent_path(canon)
        p, err = self._physical(path)
        if err or p is None:
            return LsResult(error=err or "Error: path not found.")
        if not p.exists():
            return LsResult(error=f"Error: directory '{path}' not found.")
        if not p.is_dir():
            return LsResult(error=f"Error: '{path}' is not a directory.")
        entries: list[FileInfo] = []
        try:
            for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if under_deepx:
                    ap = self._agent_path_for_deepx_file(child)
                else:
                    if _under_data_root(child, self._data_root):
                        continue
                    ap = self._agent_path_for_file(child)
                if ap is None:
                    continue
                try:
                    if child.is_dir():
                        entries.append(FileInfo(path=ap, is_dir=True))
                    else:
                        st = child.stat()
                        ts = datetime.fromtimestamp(
                            st.st_mtime, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M")
                        entries.append(
                            FileInfo(
                                path=ap, is_dir=False, size=st.st_size, modified_at=ts
                            )
                        )
                except OSError:
                    continue
        except OSError as e:
            return LsResult(error=f"Error: {e}")
        return LsResult(entries=entries)

    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        _ = session_id
        p, err = self._physical(file_path)
        if err or p is None:
            return ReadResult(error=err or f"Error: '{file_path}' not found.")
        if not p.is_file():
            return ReadResult(error=f"Error: '{file_path}' not found.")
        try:
            raw = _read_file_bytes(p)
        except OSError as e:
            return ReadResult(error=f"Error: {e}")
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        if total > 0 and offset >= total:
            return ReadResult(
                content=(
                    f"No lines to read at offset {offset}: file has {total} line(s) "
                    f"(valid offsets are 0–{total - 1})."
                ),
                total_lines=total,
            )
        selected = lines[offset : offset + limit]
        return ReadResult(content="\n".join(selected), total_lines=total)

    def _grep_via_ripgrep(
        self,
        base_phys: Path,
        pattern: str,
        glob_pat: str | None,
        *,
        under_deepx: bool = False,
    ) -> list[GrepMatch] | None:
        if not shutil.which("rg"):
            return None
        cmd: list[str] = [
            "rg",
            "--no-heading",
            "-n",
            "--color",
            "never",
            "-F",
            pattern,
            "--max-filesize",
            "10M",
        ]
        if glob_pat:
            cmd.extend(["--glob", glob_pat])
        cmd.append(str(base_phys))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120.0,
                errors="replace",
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode not in (0, 1):
            return None
        out = (proc.stdout or "").strip()
        if proc.returncode == 1 and not out:
            return []
        matches: list[GrepMatch] = []
        for line in (proc.stdout or "").splitlines():
            if not line.strip():
                continue
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path_s, ln_s, content = parts
            if not ln_s.isdigit():
                continue
            try:
                raw_p = Path(path_s).expanduser()
                fp = raw_p if raw_p.is_absolute() else (self._root_dir / raw_p)
                fp = fp.resolve()
            except OSError:
                continue
            if under_deepx:
                ap = self._agent_path_for_deepx_file(fp)
            else:
                if _under_data_root(fp, self._data_root):
                    continue
                try:
                    fp.relative_to(self._root_dir.resolve())
                except ValueError:
                    continue
                ap = self._agent_path_for_file(fp)
            if ap is None:
                continue
            matches.append(GrepMatch(path=ap, line_number=int(ln_s), line=content))
        return matches

    def _grep_via_python(
        self,
        base_phys: Path,
        pattern: str,
        glob_pat: str | None,
        *,
        under_deepx: bool = False,
    ) -> GrepResult:
        candidates: list[Path] = []
        _gflags = wcglob.EXTGLOB | wcglob.GLOBSTAR | wcglob.DOTGLOB
        if base_phys.is_file():
            candidates = [base_phys]
        else:
            for f in sorted(base_phys.rglob("*")):
                if not f.is_file():
                    continue
                if not under_deepx and _under_data_root(f, self._data_root):
                    continue
                rel = f.relative_to(base_phys).as_posix()
                if glob_pat and not (
                    wc_fnmatch.fnmatch(rel, glob_pat, flags=_gflags)
                    or wc_fnmatch.fnmatch(f.name, glob_pat, flags=_gflags)
                ):
                    continue
                try:
                    if f.stat().st_size > MAX_GREP_FILE_BYTES:
                        continue
                except OSError:
                    continue
                candidates.append(f)
        matches: list[GrepMatch] = []
        for fp in candidates:
            ap = (
                self._agent_path_for_deepx_file(fp)
                if under_deepx
                else self._agent_path_for_file(fp)
            )
            if ap is None:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, ln in enumerate(text.splitlines(), start=1):
                if pattern in ln:
                    matches.append(GrepMatch(path=ap, line_number=i, line=ln))
        return GrepResult(matches=matches)

    def grep(
        self,
        session_id: str,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        _ = session_id
        base_agent = _norm_agent_path(path or "/")
        under_deepx = self._is_deepx_agent_path(base_agent)
        base_phys, err = self._physical(base_agent)
        if err or base_phys is None:
            return GrepResult(error=err or f"Error: path '{base_agent}' not found.")
        if not base_phys.exists():
            return GrepResult(error=f"Error: path '{base_agent}' not found.")
        rg_matches = self._grep_via_ripgrep(
            base_phys, pattern, glob, under_deepx=under_deepx
        )
        if rg_matches is not None:
            return GrepResult(matches=rg_matches)
        return self._grep_via_python(
            base_phys, pattern, glob, under_deepx=under_deepx
        )

    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        _ = session_id
        base_agent = _norm_agent_path(path)
        under_deepx = self._is_deepx_agent_path(base_agent)
        base, err = self._physical(base_agent)
        if err or base is None:
            return GlobResult(error=err or f"Error: path '{base_agent}' not found.")
        if not base.exists():
            return GlobResult(error=f"Error: path '{base_agent}' not found.")
        files: list[FileInfo] = []
        try:
            _flags = wcglob.GLOBSTAR | wcglob.EXTGLOB | wcglob.BRACE | wcglob.DOTGLOB
            rel_hits = wcglob.glob(
                pattern,
                root_dir=str(base),
                flags=_flags,
                limit=5001,
            )
            for rel in sorted(rel_hits)[:500]:
                f = (base / rel).resolve()
                try:
                    f.relative_to(base.resolve())
                except ValueError:
                    continue
                if not under_deepx and _under_data_root(f, self._data_root):
                    continue
                if not f.is_file():
                    continue
                ap = (
                    self._agent_path_for_deepx_file(f)
                    if under_deepx
                    else self._agent_path_for_file(f)
                )
                if ap is None:
                    continue
                st = f.stat()
                ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
                files.append(
                    FileInfo(path=ap, is_dir=False, size=st.st_size, modified_at=ts)
                )
        except OSError as e:
            return GlobResult(error=f"Error: {e}")
        files.sort(key=lambda x: x.path)
        return GlobResult(files=files)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        _ = session_id
        canonical = self._canon(file_path)
        p, err = self._physical(file_path)
        if err or p is None:
            return WriteResult(error=err or "Cannot write: invalid path.")
        allow_replace = self._is_deepx_agent_path(
            canonical
        ) or canonical.startswith(OUTPUTS_PREFIX + "/")
        if p.exists() and not allow_replace:
            return WriteResult(
                error=f"Cannot write to {canonical} because it already exists."
            )
        try:
            _write_file_bytes(p, content.encode("utf-8"))
        except OSError as e:
            return WriteResult(error=f"Error: {e}")
        return WriteResult(path=canonical, files_update=None)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        _ = session_id
        p, err = self._physical(file_path)
        if err or p is None:
            return EditResult(error=err or f"Error: '{file_path}' not found.")
        if not p.is_file():
            return EditResult(error=f"Error: '{file_path}' not found.")
        try:
            content = _read_file_bytes(p).decode("utf-8", errors="replace")
        except OSError as e:
            return EditResult(error=f"Error: {e}")
        count = content.count(old_string)
        if count == 0:
            return EditResult(error=f"Error: string not found in '{file_path}'.")
        if not replace_all and count > 1:
            return EditResult(
                error=(
                    f"Error: String appears {count} times in file. "
                    "Use replace_all=True or provide a more specific old_string."
                )
            )
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        try:
            _write_file_bytes(p, new_content.encode("utf-8"))
        except OSError as e:
            return EditResult(error=f"Error: {e}")
        return EditResult(path=file_path, occurrences=count if replace_all else 1)

    def execute(
        self,
        session_id: str,
        command: str,
        *,
        timeout: float = 120.0,
    ) -> str:
        _ = session_id, command, timeout
        return (
            "Shell execution is not available on FilesystemBackend. "
            "Use LocalShellBackend (or another backend that implements execute) for the execute tool."
        )
