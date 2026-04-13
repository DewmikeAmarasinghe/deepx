from deepx.backends.composite import CompositeBackend
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory import InMemoryBackend
from deepx.backends.protocol import (
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

__all__ = [
    "BackendProtocol",
    "ReadResult",
    "WriteResult",
    "EditResult",
    "LsResult",
    "GlobResult",
    "GrepResult",
    "GrepMatch",
    "FileInfo",
    "FilesystemBackend",
    "InMemoryBackend",
    "CompositeBackend",
]
