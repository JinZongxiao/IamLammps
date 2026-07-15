"""Immutable, traceable storage for local LAMMPS executions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from .lammps_runner import LammpsRunResult, run_lammps
from .workspace import Workspace


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The stable, workspace-relative record of one execution."""

    schema_version: int
    run_id: str
    job: str
    status: RunStatus
    created_at: str
    completed_at: str
    source_input: str
    input_sha256: str
    artifact_dir: str
    workspace_dir: str
    stdout_path: str
    stderr_path: str
    lammps_log_path: str | None
    manifest_path: str
    return_code: int | None
    wall_time: float
    command: tuple[str, ...]
    output_files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["command"] = list(self.command)
        payload["output_files"] = list(self.output_files)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RunManifest:
        return cls(
            schema_version=_manifest_int(payload, "schema_version"),
            run_id=_manifest_string(payload, "run_id"),
            job=_manifest_string(payload, "job"),
            status=RunStatus(_manifest_string(payload, "status")),
            created_at=_manifest_string(payload, "created_at"),
            completed_at=_manifest_string(payload, "completed_at"),
            source_input=_manifest_string(payload, "source_input"),
            input_sha256=_manifest_string(payload, "input_sha256"),
            artifact_dir=_manifest_string(payload, "artifact_dir"),
            workspace_dir=_manifest_string(payload, "workspace_dir"),
            stdout_path=_manifest_string(payload, "stdout_path"),
            stderr_path=_manifest_string(payload, "stderr_path"),
            lammps_log_path=_manifest_optional_string(payload, "lammps_log_path"),
            manifest_path=_manifest_string(payload, "manifest_path"),
            return_code=_manifest_optional_int(payload, "return_code"),
            wall_time=_manifest_float(payload, "wall_time"),
            command=_manifest_string_tuple(payload, "command"),
            output_files=_manifest_string_tuple(payload, "output_files"),
        )


Runner = Callable[..., LammpsRunResult]
_JOB_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RUN_ID_RE = re.compile(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{12}")


def _manifest_string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"manifest field {key!r} must be a string")
    return value


def _manifest_optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"manifest field {key!r} must be a string or null")
    return value


def _manifest_int(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest field {key!r} must be an integer")
    return value


def _manifest_optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest field {key!r} must be an integer or null")
    return value


def _manifest_float(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"manifest field {key!r} must be numeric")
    return float(value)


def _manifest_string_tuple(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"manifest field {key!r} must be a list of strings")
    return tuple(value)


class RunArtifactStore:
    """Create and inspect immutable execution snapshots below ``runs/``."""

    def __init__(self, workspace_root: str | Path) -> None:
        root = Path(workspace_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        self.root = root

    def execute(
        self,
        input_path: str | Path,
        *,
        command: str,
        expected_sha256: str | None = None,
        timeout: int | None = None,
        mpi_np: int | None = None,
        runner: Runner = run_lammps,
        environment: dict[str, str] | None = None,
    ) -> RunManifest:
        source, job, relative_entry = self._resolve_source_input(input_path)
        if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
        source_sha256 = _sha256_file(source)
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise ValueError("LAMMPS input changed after it was reviewed")

        run_id = self._new_run_id()
        created_at = _utc_now()
        executions_dir = self.root / "runs" / job / "executions"
        _reject_symlink_components(self.root, executions_dir)
        executions_dir.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(self.root, executions_dir)
        if not executions_dir.resolve(strict=True).is_relative_to(self.root):
            raise ValueError("executions directory escapes workspace")
        execution_dir = executions_dir / run_id
        workspace_dir = execution_dir / "workspace"
        execution_dir.mkdir(parents=True, exist_ok=False)
        try:
            self._copy_inputs(self.root / "runs" / job / "inputs", workspace_dir)
            snapshot_input = workspace_dir / relative_entry
            snapshot_sha256 = _sha256_file(snapshot_input)
            if snapshot_sha256 != source_sha256:
                raise RuntimeError("input changed while the immutable snapshot was created")
            for copied_file in workspace_dir.rglob("*"):
                if copied_file.is_file():
                    copied_file.chmod(copied_file.stat().st_mode & ~0o222)
        except Exception:
            shutil.rmtree(execution_dir, ignore_errors=True)
            raise

        relative_execution = execution_dir.relative_to(self.root).as_posix()
        relative_workspace = workspace_dir.relative_to(self.root).as_posix()
        manifest = RunManifest(
            schema_version=1,
            run_id=run_id,
            job=job,
            status=RunStatus.FAILED,
            created_at=created_at,
            completed_at=created_at,
            source_input=source.relative_to(self.root).as_posix(),
            input_sha256=snapshot_sha256,
            artifact_dir=relative_execution,
            workspace_dir=relative_workspace,
            stdout_path=f"{relative_execution}/stdout.log",
            stderr_path=f"{relative_execution}/stderr.log",
            lammps_log_path=None,
            manifest_path=f"{relative_execution}/manifest.json",
            return_code=None,
            wall_time=0.0,
            command=(),
            output_files=(),
        )

        try:
            result = runner(
                command=command,
                input_file=snapshot_input,
                work_dir=workspace_dir,
                timeout=timeout,
                mpi_np=mpi_np,
                output_dir=execution_dir,
                persist_result=False,
                environment=environment,
            )
        except Exception as exc:
            (execution_dir / "stderr.log").write_text(
                f"Failed to launch LAMMPS: {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            (execution_dir / "stdout.log").touch()
            result = LammpsRunResult(
                command=[],
                return_code=-1,
                stdout_path=str(execution_dir / "stdout.log"),
                stderr_path=str(execution_dir / "stderr.log"),
            )

        status = (
            RunStatus.TIMED_OUT
            if result.timed_out
            else RunStatus.SUCCEEDED
            if result.return_code == 0
            else RunStatus.FAILED
        )
        output_files = tuple(
            sorted(
                f"{relative_workspace}/{path}"
                for path in _validated_output_files(workspace_dir, result.output_files)
            )
        )
        log_path = workspace_dir / "log.lammps"
        manifest = replace(
            manifest,
            status=status,
            completed_at=_utc_now(),
            lammps_log_path=(
                log_path.relative_to(self.root).as_posix() if log_path.is_file() else None
            ),
            return_code=result.return_code,
            wall_time=result.wall_time,
            command=tuple(_workspace_relative_argument(self.root, item) for item in result.command),
            output_files=output_files,
        )
        self._persist_manifest(manifest)
        self._update_latest(manifest)
        return manifest

    def list_runs(self, job: str) -> list[RunManifest]:
        self._validate_job(job)
        executions = self.root / "runs" / job / "executions"
        if not executions.exists():
            return []
        _reject_symlink_components(self.root, executions)
        if executions.is_symlink() or not executions.is_dir():
            raise ValueError(f"unsafe executions directory for job {job!r}")
        manifests: list[RunManifest] = []
        for directory in sorted(executions.iterdir(), reverse=True):
            if directory.is_symlink() or not directory.is_dir():
                continue
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file() and not manifest_path.is_symlink():
                manifests.append(self._read_manifest(manifest_path))
        return sorted(manifests, key=lambda item: item.created_at, reverse=True)

    def inspect_run(self, run_id: str) -> RunManifest:
        if _RUN_ID_RE.fullmatch(run_id) is None:
            raise ValueError("invalid run_id")
        matches: list[Path] = []
        runs = self.root / "runs"
        if runs.exists():
            _reject_symlink_components(self.root, runs)
            for job_dir in runs.iterdir():
                if job_dir.is_symlink() or not job_dir.is_dir():
                    continue
                candidate = job_dir / "executions" / run_id / "manifest.json"
                if candidate.is_file() and not candidate.is_symlink():
                    matches.append(candidate)
        if not matches:
            raise FileNotFoundError(f"unknown run_id: {run_id}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous run_id: {run_id}")
        return self._read_manifest(matches[0])

    def resolve_manifest_path(self, path: str) -> Path:
        """Resolve one path from a trusted manifest and keep it below this root."""
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest path must be workspace-relative")
        resolved = (self.root / relative).resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ValueError("manifest path escapes workspace")
        return resolved

    def _resolve_source_input(self, input_path: str | Path) -> tuple[Path, str, Path]:
        relative = Path(input_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input_path must be workspace-relative")
        if len(relative.parts) < 4 or relative.parts[0] != "runs" or relative.parts[2] != "inputs":
            raise ValueError("LAMMPS input must be below runs/<job>/inputs/")
        job = relative.parts[1]
        self._validate_job(job)
        lexical = self.root / relative
        _reject_symlink_components(self.root, lexical)
        source = lexical.resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(self.root):
            raise FileNotFoundError(relative)
        return source, job, Path(*relative.parts[3:])

    def _copy_inputs(self, inputs_dir: Path, destination: Path) -> None:
        relative_inputs = inputs_dir.relative_to(self.root)
        source_fd = _open_directory_beneath(self.root, relative_inputs)
        destination.mkdir(parents=True, exist_ok=False)
        destination_fd = _open_directory_beneath(
            self.root, destination.relative_to(self.root)
        )
        try:
            _copy_tree_fd(source_fd, destination_fd)
        finally:
            os.close(source_fd)
            os.close(destination_fd)

    def _persist_manifest(self, manifest: RunManifest) -> None:
        path = self.resolve_manifest_path(manifest.manifest_path)
        _safe_json(self.root, path, manifest.to_dict())

    def _update_latest(self, manifest: RunManifest) -> None:
        path = self.root / "runs" / manifest.job / "latest.json"
        _safe_json(
            self.root,
            path,
            {"run_id": manifest.run_id, "manifest": manifest.manifest_path},
        )

    def _read_manifest(self, path: Path) -> RunManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid run manifest: {path}")
        manifest = RunManifest.from_dict(payload)
        expected_artifact = f"runs/{manifest.job}/executions/{manifest.run_id}"
        expected_workspace = f"{expected_artifact}/workspace"
        expected_paths = {
            "artifact_dir": expected_artifact,
            "workspace_dir": expected_workspace,
            "stdout_path": f"{expected_artifact}/stdout.log",
            "stderr_path": f"{expected_artifact}/stderr.log",
            "manifest_path": f"{expected_artifact}/manifest.json",
        }
        for field, expected in expected_paths.items():
            if getattr(manifest, field) != expected:
                raise ValueError(f"manifest {field} mismatch: {path}")
        if self.resolve_manifest_path(manifest.manifest_path) != path.resolve():
            raise ValueError(f"manifest path mismatch: {path}")
        allowed_output_prefix = f"{expected_workspace}/"
        if any(not output.startswith(allowed_output_prefix) for output in manifest.output_files):
            raise ValueError(f"manifest output path mismatch: {path}")
        if manifest.lammps_log_path is not None and not manifest.lammps_log_path.startswith(
            allowed_output_prefix
        ):
            raise ValueError(f"manifest LAMMPS log path mismatch: {path}")
        if not manifest.source_input.startswith(f"runs/{manifest.job}/inputs/"):
            raise ValueError(f"manifest source input mismatch: {path}")
        for manifest_path in (
            manifest.artifact_dir,
            manifest.workspace_dir,
            manifest.stdout_path,
            manifest.stderr_path,
            manifest.manifest_path,
            *manifest.output_files,
        ):
            self.resolve_manifest_path(manifest_path)
        return manifest

    def _new_run_id(self) -> str:
        while True:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            run_id = f"{timestamp}-{uuid.uuid4().hex[:12]}"
            if not any((self.root / "runs").glob(f"*/executions/{run_id}")):
                return run_id

    @staticmethod
    def _validate_job(job: str) -> None:
        if _JOB_RE.fullmatch(job) is None or job in {".", ".."}:
            raise ValueError("invalid job name")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_directory_beneath(root: Path, relative: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        for part in relative.parts:
            child = os.open(part, flags | nofollow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _copy_tree_fd(source_fd: int, destination_fd: int) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    for name in sorted(os.listdir(source_fd)):
        details = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISDIR(details.st_mode):
            os.mkdir(name, details.st_mode & 0o777, dir_fd=destination_fd)
            child_source = os.open(
                name, os.O_RDONLY | directory_flag | nofollow, dir_fd=source_fd
            )
            child_destination = os.open(
                name,
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=destination_fd,
            )
            try:
                _copy_tree_fd(child_source, child_destination)
            finally:
                os.close(child_source)
                os.close(child_destination)
            continue
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("input tree must contain only regular files and directories")
        source_file = os.open(name, os.O_RDONLY | nofollow, dir_fd=source_fd)
        try:
            opened = os.fstat(source_file)
            if (opened.st_dev, opened.st_ino) != (details.st_dev, details.st_ino):
                raise ValueError("input file changed while its snapshot was created")
            destination_file = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                details.st_mode & 0o555,
                dir_fd=destination_fd,
            )
            try:
                while chunk := os.read(source_file, 1024 * 1024):
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_file, view)
                        view = view[written:]
                os.fsync(destination_file)
                os.fchmod(destination_file, details.st_mode & 0o555)
            finally:
                os.close(destination_file)
        finally:
            os.close(source_file)


def _reject_symlink_components(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinks are not allowed in run paths: {relative}")


def _safe_json(root: Path, path: Path, payload: object) -> None:
    workspace = Workspace(root)
    relative = path.relative_to(root)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        current = workspace.read_text(relative)
    except FileNotFoundError:
        workspace.write_text(relative, content)
    else:
        workspace.write_text(relative, content, expected_sha256=current.sha256)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _workspace_relative_argument(root: Path, argument: str) -> str:
    try:
        path = Path(argument)
        if path.is_absolute() and path.is_relative_to(root):
            return path.relative_to(root).as_posix()
    except (OSError, ValueError):
        pass
    return argument


def _validated_output_files(workspace_dir: Path, output_files: list[str]) -> list[str]:
    validated: list[str] = []
    for value in output_files:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError(f"runner returned an unsafe output path: {value}")
        candidate = workspace_dir / relative
        _reject_symlink_components(workspace_dir, candidate)
        if not candidate.is_file():
            raise ValueError(f"runner reported a missing output file: {value}")
        validated.append(relative.as_posix())
    return validated
