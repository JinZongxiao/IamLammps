from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict, field
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
) -> LammpsRunResult:
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"

    input_file = input_file.resolve()
    cmd: list[str] = []
    if mpi_np is not None and mpi_np > 1:
        cmd = ["mpirun", "-np", str(mpi_np)]
    # Split command string into list of arguments (e.g., "lmp -k on -sf kk" -> ["lmp", "-k", "on", "-sf", "kk"])
    cmd.extend(shlex.split(command))
    cmd.extend(["-in", str(input_file)])

    # Snapshot existing files so we can detect new outputs.
    files_before = set(os.listdir(work_dir))

    timed_out = False
    t0 = time.monotonic()

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        try:
            process = subprocess.run(
                cmd, stdout=out, stderr=err, cwd=work_dir, check=False,
                timeout=timeout,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1

    wall_time = time.monotonic() - t0

    # Detect new output files (dump, restart, data, etc.).
    files_after = set(os.listdir(work_dir))
    new_files = sorted(files_after - files_before - {"stdout.log", "stderr.log", "result.json"})

    result = LammpsRunResult(
        command=cmd,
        return_code=returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        wall_time=round(wall_time, 2),
        timed_out=timed_out,
        output_files=new_files,
    )

    with (work_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    return result
