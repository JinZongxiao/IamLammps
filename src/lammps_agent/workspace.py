"""Safe, bounded access to text files in a project workspace."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


class WorkspaceError(ValueError):
    """Base class for workspace access errors."""


class UnsafePathError(WorkspaceError):
    """Raised when a path could escape the workspace."""


class ProtectedPathError(WorkspaceError):
    """Raised when a sensitive path is requested."""


class FileTooLargeError(WorkspaceError):
    """Raised when a file exceeds the configured text size limit."""


class BinaryFileError(WorkspaceError):
    """Raised when a file is not UTF-8 text."""


class WorkspaceConflictError(WorkspaceError):
    """Raised when a file changed since it was read."""


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    """Metadata returned after listing or changing a workspace file."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class WorkspaceText(WorkspaceFile):
    """A bounded text file and its metadata."""

    content: str


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PRIVATE_KEY_NAMES = {
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
_PRIVATE_KEY_SUFFIXES = {".key", ".p12", ".pem", ".pfx", ".pkcs12"}
_SENSITIVE_DIRECTORY_NAMES = {
    ".aws",
    ".azure",
    ".docker",
    ".gnupg",
    ".kube",
    ".password-store",
}
_SENSITIVE_FILE_NAMES = {
    ".authinfo",
    ".authinfo.gpg",
    ".dockercfg",
    ".envrc",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".vault-token",
    "auth.json",
    "credentials.json",
    "secrets.json",
}


class Workspace:
    """Expose a narrow text-file API rooted at one project directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_text_bytes: int = 1_000_000,
        max_list_files: int = 200,
        max_scan_entries: int = 5_000,
    ) -> None:
        if max_text_bytes <= 0 or max_list_files <= 0 or max_scan_entries <= 0:
            raise ValueError("workspace limits must be positive")
        resolved_root = Path(root).expanduser().resolve(strict=True)
        if not resolved_root.is_dir():
            raise NotADirectoryError(resolved_root)
        self.root = resolved_root
        self.max_text_bytes = max_text_bytes
        self.max_list_files = max_list_files
        self.max_scan_entries = max_scan_entries

    def resolve(self, path: str | Path) -> Path:
        """Resolve a relative, non-sensitive path without leaving the root."""

        relative = Path(path)
        if relative.is_absolute():
            raise UnsafePathError("absolute paths are not allowed")
        if any(part == ".." for part in relative.parts):
            raise UnsafePathError("parent path components are not allowed")
        self._assert_not_protected(relative)

        candidate = (self.root / relative).resolve(strict=False)
        try:
            resolved_relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise UnsafePathError("path escapes the workspace") from exc
        self._assert_not_protected(resolved_relative)
        return candidate

    def relative_path(self, path: str | Path) -> Path:
        """Validate and return a lexical workspace-relative path.

        Unlike :meth:`resolve`, this helper never follows a filesystem entry.
        It is the starting point for descriptor-relative safe file operations.
        """

        relative = Path(path)
        if relative.is_absolute():
            raise UnsafePathError("absolute paths are not allowed")
        if not relative.parts or relative == Path("."):
            raise UnsafePathError("a file path is required")
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise UnsafePathError("dot and parent path components are not allowed")
        self._assert_not_protected(relative)
        return relative

    def list_text_files(
        self,
        directory: str | Path = ".",
        *,
        max_files: int | None = None,
    ) -> list[WorkspaceFile]:
        """List bounded UTF-8 text files below a workspace directory."""

        limit = self.max_list_files if max_files is None else max_files
        if limit <= 0 or limit > self.max_list_files:
            raise ValueError(f"max_files must be between 1 and {self.max_list_files}")
        base = self.resolve(directory)
        if not base.is_dir():
            raise NotADirectoryError(base)

        results: list[WorkspaceFile] = []
        seen: set[Path] = set()
        scanned = 0
        for current, directory_names, file_names in os.walk(base, followlinks=False):
            directory_names.sort()
            file_names.sort()
            safe_directories: list[str] = []
            for name in directory_names:
                scanned += 1
                if scanned > self.max_scan_entries:
                    break
                child = Path(current, name)
                if child.is_symlink():
                    continue
                try:
                    self.resolve(child.relative_to(self.root))
                except (ProtectedPathError, UnsafePathError):
                    continue
                safe_directories.append(name)
            directory_names[:] = safe_directories
            if scanned > self.max_scan_entries:
                break

            for name in file_names:
                scanned += 1
                if scanned > self.max_scan_entries or len(results) >= limit:
                    break
                lexical_path = Path(current, name)
                try:
                    relative = self.relative_path(lexical_path.relative_to(self.root))
                except (ProtectedPathError, UnsafePathError):
                    continue
                if relative in seen:
                    continue
                try:
                    raw = self._read_relative(relative, self.max_text_bytes)
                    self._decode_text(raw)
                except (
                    BinaryFileError,
                    FileTooLargeError,
                    UnsafePathError,
                    OSError,
                ):
                    continue
                seen.add(relative)
                results.append(self._relative_metadata(relative, raw))
            if scanned > self.max_scan_entries or len(results) >= limit:
                break
        return results

    def read_text(
        self,
        path: str | Path,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceText:
        """Read bounded UTF-8 text with a digest for later safe updates."""

        limit = self.max_text_bytes if max_bytes is None else max_bytes
        if limit <= 0 or limit > self.max_text_bytes:
            raise ValueError(f"max_bytes must be between 1 and {self.max_text_bytes}")
        relative = self.relative_path(path)
        raw = self._read_relative(relative, limit)
        content = self._decode_text(raw)
        metadata = self._relative_metadata(relative, raw)
        return WorkspaceText(
            path=metadata.path,
            sha256=metadata.sha256,
            size=metadata.size,
            content=content,
        )

    def write_text(
        self,
        path: str | Path,
        content: str,
        *,
        expected_sha256: str | None = None,
    ) -> WorkspaceFile:
        """Atomically create a file or replace the exact version previously read."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        raw = content.encode("utf-8")
        if len(raw) > self.max_text_bytes:
            raise FileTooLargeError(
                f"content is {len(raw)} bytes; limit is {self.max_text_bytes}"
            )
        if expected_sha256 is not None:
            self._validate_sha256(expected_sha256)

        relative = self.relative_path(path)
        with self._open_parent(relative, create=True) as (parent_fd, name):
            existing_mode: int | None = None
            if expected_sha256 is None:
                try:
                    existing_fd = self._open_file(parent_fd, name)
                except FileNotFoundError:
                    pass
                else:
                    os.close(existing_fd)
                    raise FileExistsError(f"refusing to overwrite existing file: {path}")
            else:
                try:
                    existing_fd = self._open_file(parent_fd, name)
                except FileNotFoundError as exc:
                    raise WorkspaceConflictError(f"file does not exist: {path}") from exc
                try:
                    existing = self._read_fd(existing_fd, self.max_text_bytes)
                    actual_sha256 = hashlib.sha256(existing).hexdigest()
                    if actual_sha256 != expected_sha256:
                        raise WorkspaceConflictError(
                            f"file changed: expected {expected_sha256}, got {actual_sha256}"
                        )
                    existing_mode = os.fstat(existing_fd).st_mode & 0o777
                finally:
                    os.close(existing_fd)

            temporary_name = f".{name}.{uuid4().hex}.tmp"
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
                existing_mode if existing_mode is not None else 0o644,
                dir_fd=parent_fd,
            )
            try:
                self._write_all(temporary_fd, raw)
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)
            try:
                if expected_sha256 is None:
                    try:
                        os.link(
                            temporary_name,
                            name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise FileExistsError(
                            f"refusing to overwrite existing file: {path}"
                        ) from exc
                else:
                    # The parent directory is pinned by descriptor, and rename
                    # replaces a leaf symlink rather than following it.
                    current_fd = self._open_file(parent_fd, name)
                    try:
                        current = self._read_fd(current_fd, self.max_text_bytes)
                    finally:
                        os.close(current_fd)
                    current_sha256 = hashlib.sha256(current).hexdigest()
                    if current_sha256 != expected_sha256:
                        raise WorkspaceConflictError(
                            f"file changed: expected {expected_sha256}, got {current_sha256}"
                        )
                    os.replace(
                        temporary_name,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                os.fsync(parent_fd)
            finally:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            return self._relative_metadata(relative, raw)

    def replace_text(
        self,
        path: str | Path,
        old: str,
        new: str,
        *,
        expected_sha256: str,
    ) -> WorkspaceFile:
        """Replace one exact occurrence in the expected version of a text file."""

        self._validate_sha256(expected_sha256)
        if not old:
            raise ValueError("old text must not be empty")
        current = self.read_text(path)
        if current.sha256 != expected_sha256:
            raise WorkspaceConflictError(
                f"file changed: expected {expected_sha256}, got {current.sha256}"
            )
        occurrences = current.content.count(old)
        if occurrences != 1:
            raise WorkspaceConflictError(
                f"old text must occur exactly once; found {occurrences} occurrences"
            )
        updated = current.content.replace(old, new, 1)
        return self.write_text(
            path,
            updated,
            expected_sha256=expected_sha256,
        )

    def _metadata(self, path: Path, raw: bytes) -> WorkspaceFile:
        return WorkspaceFile(
            path=path.relative_to(self.root).as_posix(),
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )

    @staticmethod
    def _relative_metadata(path: Path, raw: bytes) -> WorkspaceFile:
        return WorkspaceFile(
            path=path.as_posix(),
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryFileError("file is not UTF-8 text") from exc
        if "\x00" in content:
            raise BinaryFileError("file contains NUL bytes")
        return content

    @staticmethod
    def _read_bytes(path: Path, limit: int) -> bytes:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > limit:
            raise FileTooLargeError(f"file exceeds the {limit}-byte limit")
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise FileTooLargeError(f"file exceeds the {limit}-byte limit")
        return raw

    def _read_relative(self, path: Path, limit: int) -> bytes:
        with self._open_parent(path) as (parent_fd, name):
            descriptor = self._open_file(parent_fd, name)
            try:
                return self._read_fd(descriptor, limit)
            finally:
                os.close(descriptor)

    @contextmanager
    def _open_parent(
        self, path: Path, *, create: bool = False
    ) -> Iterator[tuple[int, str]]:
        descriptor = os.open(self.root, os.O_RDONLY | _O_DIRECTORY)
        try:
            for part in path.parts[:-1]:
                try:
                    child = os.open(
                        part,
                        os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = os.open(
                        part,
                        os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise UnsafePathError(
                            "workspace directory components must not be symbolic links"
                        ) from exc
                    raise
                os.close(descriptor)
                descriptor = child
            yield descriptor, path.name
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_file(parent_fd: int, name: str) -> int:
        try:
            descriptor = os.open(name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise UnsafePathError(
                    "workspace files must not be symbolic links"
                ) from exc
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise UnsafePathError("workspace entry must be a regular file")
        return descriptor

    @staticmethod
    def _read_fd(descriptor: int, limit: int) -> bytes:
        details = os.fstat(descriptor)
        if details.st_size > limit:
            raise FileTooLargeError(f"file exceeds the {limit}-byte limit")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise FileTooLargeError(f"file exceeds the {limit}-byte limit")
        return raw

    @staticmethod
    def _write_all(descriptor: int, raw: bytes) -> None:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]

    @staticmethod
    def _validate_sha256(value: str) -> None:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")

    @staticmethod
    def _assert_not_protected(path: Path) -> None:
        for part in path.parts:
            lowered = part.lower()
            if (
                lowered in {
                    ".git",
                    ".ssh",
                    ".env",
                    ".iamlammps",
                    *_SENSITIVE_DIRECTORY_NAMES,
                }
                or lowered.startswith(".env.")
            ):
                raise ProtectedPathError(f"protected path: {path}")
        name = path.name.lower()
        if (
            name == "agents.md"
            or
            name in _SENSITIVE_FILE_NAMES
            or
            name in _PRIVATE_KEY_NAMES
            or Path(name).suffix in _PRIVATE_KEY_SUFFIXES
            or "private_key" in name
            or "private-key" in name
        ):
            raise ProtectedPathError(f"private key path is protected: {path}")


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
