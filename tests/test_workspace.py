from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from lammps_agent.workspace import (
    BinaryFileError,
    FileTooLargeError,
    ProtectedPathError,
    UnsafePathError,
    Workspace,
    WorkspaceConflictError,
)


def test_resolve_rejects_absolute_parent_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    workspace = Workspace(root)

    with pytest.raises(UnsafePathError):
        workspace.resolve(outside)
    with pytest.raises(UnsafePathError):
        workspace.resolve("../outside")
    with pytest.raises(UnsafePathError):
        workspace.resolve("escape/file.txt")


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".git/config",
        ".iamlammps/mcp.json",
        ".iamlammps/skills/custom/SKILL.md",
        ".ssh/custom_identity",
        "AGENTS.md",
        "id_rsa",
        "keys/id_ed25519",
        "keys/server.pem",
        "keys/private_key.txt",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".git-credentials",
        ".aws/credentials",
        ".docker/config.json",
        ".kube/config",
        "credentials.json",
        "secrets.json",
    ],
)
def test_protected_files_cannot_be_read_or_written(tmp_path: Path, path: str) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(ProtectedPathError):
        workspace.read_text(path)
    with pytest.raises(ProtectedPathError):
        workspace.write_text(path, "secret")


def test_read_text_is_bounded_and_returns_digest(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello\n", encoding="utf-8")
    workspace = Workspace(tmp_path, max_text_bytes=8)

    result = workspace.read_text("notes.txt")

    assert result.path == "notes.txt"
    assert result.content == "hello\n"
    assert result.size == 6
    assert result.sha256 == hashlib.sha256(b"hello\n").hexdigest()

    path.write_text("123456789", encoding="utf-8")
    with pytest.raises(FileTooLargeError):
        workspace.read_text("notes.txt")


def test_read_text_rejects_binary_content(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"abc\x00def")

    with pytest.raises(BinaryFileError):
        Workspace(tmp_path).read_text("binary.dat")


def test_list_text_files_is_safe_bounded_and_deterministic(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"\xff")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("secret", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "outside.txt").write_text("outside", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside / "outside.txt")
    workspace = Workspace(tmp_path, max_list_files=2)

    results = workspace.list_text_files()

    assert [result.path for result in results] == ["a.txt", "b.txt"]
    assert [result.size for result in results] == [1, 1]
    assert all(len(result.sha256) == 64 for result in results)
    with pytest.raises(ValueError):
        workspace.list_text_files(max_files=3)


def test_write_text_creates_parents_atomically_and_does_not_overwrite(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)

    result = workspace.write_text("new/notes.txt", "first")

    assert result.path == "new/notes.txt"
    assert result.size == 5
    assert result.sha256 == hashlib.sha256(b"first").hexdigest()
    assert (tmp_path / "new" / "notes.txt").read_text(encoding="utf-8") == "first"
    assert not list((tmp_path / "new").glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        workspace.write_text("new/notes.txt", "second")
    assert (tmp_path / "new" / "notes.txt").read_text(encoding="utf-8") == "first"


def test_write_text_requires_matching_sha_to_replace(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o640)
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceConflictError):
        workspace.write_text("notes.txt", "new", expected_sha256="0" * 64)

    original = workspace.read_text("notes.txt")
    result = workspace.write_text(
        "notes.txt",
        "new",
        expected_sha256=original.sha256,
    )

    assert path.read_text(encoding="utf-8") == "new"
    assert result.sha256 == hashlib.sha256(b"new").hexdigest()
    assert os.stat(path).st_mode & 0o777 == 0o640


def test_replace_text_requires_current_sha_and_one_exact_match(tmp_path: Path) -> None:
    path = tmp_path / "code.py"
    path.write_text("value = 1\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    current = workspace.read_text("code.py")

    result = workspace.replace_text(
        "code.py",
        "value = 1",
        "value = 2",
        expected_sha256=current.sha256,
    )

    assert path.read_text(encoding="utf-8") == "value = 2\n"
    assert result.size == len("value = 2\n")

    with pytest.raises(WorkspaceConflictError):
        workspace.replace_text(
            "code.py",
            "value = 2",
            "value = 3",
            expected_sha256=current.sha256,
        )

    duplicate = "x\nx\n"
    path.write_text(duplicate, encoding="utf-8")
    duplicate_sha = hashlib.sha256(duplicate.encode()).hexdigest()
    with pytest.raises(WorkspaceConflictError, match="2 occurrences"):
        workspace.replace_text(
            "code.py",
            "x",
            "y",
            expected_sha256=duplicate_sha,
        )


def test_write_rejects_invalid_hash_and_oversized_content(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, max_text_bytes=4)

    with pytest.raises(ValueError, match="expected_sha256"):
        workspace.write_text("file.txt", "ok", expected_sha256="not-a-hash")
    with pytest.raises(FileTooLargeError):
        workspace.write_text("file.txt", "12345")


def test_read_and_write_never_follow_leaf_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)
    workspace = Workspace(tmp_path)

    with pytest.raises(UnsafePathError, match="symbolic links"):
        workspace.read_text("linked.txt")
    with pytest.raises(UnsafePathError, match="symbolic links"):
        workspace.write_text("linked.txt", "replace")
    with pytest.raises(UnsafePathError, match="symbolic links"):
        workspace.write_text(
            "linked.txt",
            "replace",
            expected_sha256=hashlib.sha256(b"outside").hexdigest(),
        )

    assert outside.read_text(encoding="utf-8") == "outside"


def test_write_never_follows_parent_directory_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafePathError, match="symbolic links"):
        Workspace(tmp_path).write_text("linked/output.txt", "data")

    assert not (outside / "output.txt").exists()


def test_update_rechecks_cas_immediately_before_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("reviewed", encoding="utf-8")
    workspace = Workspace(tmp_path)
    reviewed = workspace.read_text("notes.txt")
    real_open_file = workspace._open_file
    opens = 0

    def swap_before_final_check(parent_fd: int, name: str) -> int:
        nonlocal opens
        opens += 1
        if opens == 2:
            path.write_text("swapped", encoding="utf-8")
        return real_open_file(parent_fd, name)

    monkeypatch.setattr(workspace, "_open_file", swap_before_final_check)

    with pytest.raises(WorkspaceConflictError, match="file changed"):
        workspace.write_text(
            "notes.txt", "replacement", expected_sha256=reviewed.sha256
        )

    assert path.read_text(encoding="utf-8") == "swapped"
