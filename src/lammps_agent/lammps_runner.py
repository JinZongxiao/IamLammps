from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LammpsRunResult:
    command: list[str]
    return_code: int
    stdout_path: str
    stderr_path: str
    wall_time: float = 0.0
    timed_out: bool = False
    output_files: list[str] = field(default_factory=list)


def run_lammps(
    command: str,
    input_file: Path,
    work_dir: Path,
    timeout: int | None = None,
    mpi_np: int | None = None,
    *,
    output_dir: Path | None = None,
    persist_result: bool = True,
    environment: dict[str, str] | None = None,
) -> LammpsRunResult:
    work_dir = work_dir.resolve(strict=True)
    output_dir = work_dir if output_dir is None else output_dir.resolve(strict=True)
    if environment is None:
        environment = _minimal_environment()
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    input_file = input_file.resolve()
    cmd: list[str] = []
    if mpi_np is not None and mpi_np > 1:
        cmd = ["mpirun", "-np", str(mpi_np)]
    cmd.extend([command, "-in", str(input_file)])

    files_before = _relative_files(work_dir)

    timed_out = False
    t0 = time.monotonic()

    with _open_log(stdout_path) as out, _open_log(stderr_path) as err:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=out,
                stderr=err,
                cwd=work_dir,
                start_new_session=True,
                env=environment,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                raise
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
        except KeyboardInterrupt:
            _terminate_process_group(process)
            raise

    wall_time = time.monotonic() - t0

    files_after = _relative_files(work_dir)
    own_outputs = {
        path.relative_to(work_dir).as_posix()
        for path in (stdout_path, stderr_path)
        if path.is_relative_to(work_dir)
    }
    new_files = sorted(files_after - files_before - own_outputs - {"result.json"})

    result = LammpsRunResult(
        command=cmd,
        return_code=returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        wall_time=round(wall_time, 2),
        timed_out=timed_out,
        output_files=new_files,
    )

    if persist_result:
        with (output_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    return result


def _relative_files(directory: Path) -> set[str]:
    """Return regular files below *directory* without traversing symlinks."""
    files: set[str] = set()
    for current, directory_names, file_names in os.walk(directory, followlinks=False):
        directory_names[:] = [
            name for name in directory_names if not Path(current, name).is_symlink()
        ]
        for name in file_names:
            path = Path(current, name)
            if path.is_file() and not path.is_symlink():
                files.add(path.relative_to(directory).as_posix())
    return files


def _open_log(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _minimal_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate the whole MPI/LAMMPS process group, escalating if needed."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
