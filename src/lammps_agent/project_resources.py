"""Install packaged project templates and diagnose a local IamLAMMPS setup."""

from __future__ import annotations

import errno
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

_RESOURCE_ROOT = ("resources", "project")
_REQUIRED_LAMMPS_PACKAGES = frozenset({"KSPACE", "MOLECULE", "RIGID"})


@dataclass(frozen=True, slots=True)
class InitResult:
    """Files created and preserved by :func:`initialize_project`."""

    root: Path
    created: tuple[Path, ...]
    skipped: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One non-secret diagnostic result."""

    name: str
    ok: bool
    detail: str


def initialize_project(directory: str | Path = ".") -> InitResult:
    """Materialize packaged templates without replacing any existing path."""

    root = Path(directory).expanduser()
    if _lexists(root) and root.is_symlink():
        raise ValueError(f"Initialization directory must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"Initialization path is not a directory: {root}")
    root = root.resolve()

    resource_root: Any = resources.files("lammps_agent")
    for part in _RESOURCE_ROOT:
        resource_root = resource_root.joinpath(part)
    if not resource_root.is_dir():
        raise RuntimeError("IamLAMMPS project resources are missing from this installation")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    root_descriptor = os.open(root, directory_flags)
    created: list[Path] = []
    skipped: list[Path] = []
    try:
        for relative, source in _resource_files(resource_root):
            parent_descriptor = _open_parent(root_descriptor, relative.parent)
            try:
                if _entry_exists(parent_descriptor, relative.name):
                    skipped.append(relative)
                    continue
                mode = 0o600 if relative.name == ".env.example" else 0o644
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(relative.name, flags, mode, dir_fd=parent_descriptor)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(source.read_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException:
                    os.unlink(relative.name, dir_fd=parent_descriptor)
                    raise
                created.append(relative)
            finally:
                os.close(parent_descriptor)
    finally:
        os.close(root_descriptor)

    return InitResult(root=root, created=tuple(created), skipped=tuple(skipped))


def run_doctor(directory: str | Path = ".") -> tuple[DoctorCheck, ...]:
    """Check local prerequisites without printing configuration values or secrets."""

    root = Path(directory).expanduser()
    checks = [
        DoctorCheck(
            "python",
            (3, 11) <= sys.version_info[:2] <= (3, 13),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    ]
    directory_ok = root.exists() and root.is_dir() and not root.is_symlink()
    writable = directory_ok and os.access(root, os.R_OK | os.W_OK | os.X_OK)
    checks.append(
        DoctorCheck(
            "workspace",
            bool(writable),
            "readable and writable" if writable else "missing, symlinked, or not writable",
        )
    )

    settings = _doctor_settings(root)
    provider = settings.get("LLM_PROVIDER", "openai_agents").strip().casefold()
    if provider == "deepseek":
        model_ok = bool(settings.get("DEEPSEEK_API_KEY", "").strip()) and bool(
            settings.get("DEEPSEEK_MODEL", "").strip()
        )
    elif provider == "openai_agents":
        model_ok = bool(settings.get("OPENAI_API_KEY", "").strip()) and bool(
            settings.get("OPENAI_MODEL", "").strip()
        )
    else:
        model_ok = False
    checks.append(
        DoctorCheck(
            "model configuration",
            model_ok,
            f"{provider} credentials and model configured"
            if model_ok
            else "provider credentials or model are missing",
        )
    )

    checks.extend(_check_lammps(settings.get("LAMMPS_COMMAND", "lmp_mpi")))
    matplotlib_ok = importlib.util.find_spec("matplotlib") is not None
    checks.append(
        DoctorCheck(
            "matplotlib",
            matplotlib_ok,
            "available" if matplotlib_ok else "not installed",
        )
    )
    return tuple(checks)


def _resource_files(root: Any, relative: Path = Path()) -> list[tuple[Path, Any]]:
    files: list[tuple[Path, Any]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name in {".", ".."}:
            raise RuntimeError("Invalid packaged resource name")
        child_relative = relative / child.name
        if child.is_dir():
            files.extend(_resource_files(child, child_relative))
        elif child.is_file():
            files.append((child_relative, child))
    return files


def _open_parent(root_descriptor: int, relative: Path) -> int:
    descriptor = os.dup(root_descriptor)
    for part in relative.parts:
        try:
            os.mkdir(part, mode=0o755, dir_fd=descriptor)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            child_descriptor = os.open(part, flags, dir_fd=descriptor)
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    f"Initialization path component is not a real directory: {relative}"
                ) from exc
            raise
        os.close(descriptor)
        descriptor = child_descriptor
    return descriptor


def _entry_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _doctor_settings(root: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    settings: dict[str, str] = {}
    env_path = root / ".env"
    if env_path.is_file() and not env_path.is_symlink():
        settings.update(
            {key: str(value) for key, value in dotenv_values(env_path).items() if value is not None}
        )
    for key in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "LAMMPS_COMMAND",
    ):
        if key in os.environ:
            settings[key] = os.environ[key]
    return settings


def _check_lammps(command: str) -> list[DoctorCheck]:
    try:
        arguments = shlex.split(command)
    except ValueError:
        arguments = []
    executable = shutil.which(arguments[0]) if arguments else None
    if executable is None:
        return [
            DoctorCheck("lammps executable", False, "not found"),
            DoctorCheck("lammps packages", False, "not checked"),
        ]

    minimal_env = {
        key: os.environ[key]
        for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
        if key in os.environ
    }
    try:
        completed = subprocess.run(
            [executable, *arguments[1:], "-help"],
            capture_output=True,
            check=False,
            env=minimal_env,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [
            DoctorCheck("lammps executable", False, f"help check failed: {type(exc).__name__}"),
            DoctorCheck("lammps packages", False, "not checked"),
        ]

    output = f"{completed.stdout}\n{completed.stderr}"
    executable_ok = completed.returncode == 0
    installed = set(re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", output))
    missing = sorted(_REQUIRED_LAMMPS_PACKAGES - installed)
    return [
        DoctorCheck(
            "lammps executable",
            executable_ok,
            "available" if executable_ok else f"help returned {completed.returncode}",
        ),
        DoctorCheck(
            "lammps packages",
            executable_ok and not missing,
            "KSPACE, MOLECULE, and RIGID available"
            if executable_ok and not missing
            else f"missing required packages: {', '.join(missing)}",
        ),
    ]


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)
