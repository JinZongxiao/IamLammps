"""Local backend — wraps the existing ``run_lammps()`` subprocess call."""

from __future__ import annotations

import uuid
from pathlib import Path

from ..core_types import SimulationTask, SimulationResult
from ..lammps_runner import run_lammps
from .base import JobBackend


class LocalBackend(JobBackend):
    """Run LAMMPS locally via subprocess.

    ``submit()`` is **synchronous** (blocking) — the simulation runs
    immediately and the result is stored internally.  ``poll()`` always
    returns ``'completed'`` or ``'failed'``.
    """

    def __init__(self, lammps_command: str = "lmp_mpi") -> None:
        self.lammps_command = lammps_command
        self._results: dict[str, SimulationResult] = {}

    def submit(self, task: SimulationTask) -> str:
        job_id = uuid.uuid4().hex[:12]

        work_dir = task.work_dir or Path(f"runs/{job_id}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write script
        input_path = work_dir / "in.lammps"
        input_path.write_text(task.code, encoding="utf-8")

        # Delegate to existing run_lammps (unchanged)
        lammps_result = run_lammps(
            command=self.lammps_command,
            input_file=input_path,
            work_dir=work_dir,
            timeout=task.resources.get("timeout"),
            mpi_np=task.resources.get("mpi_np"),
        )

        success = lammps_result.return_code == 0
        sim_result = SimulationResult(
            success=success,
            outputs={
                "stdout_path": lammps_result.stdout_path,
                "stderr_path": lammps_result.stderr_path,
                "output_files": lammps_result.output_files,
                "work_dir": str(work_dir),
            },
            timing={
                "wall_time": lammps_result.wall_time,
                "timed_out": lammps_result.timed_out,
            },
            errors=[] if success else [f"return_code={lammps_result.return_code}"],
            raw_result=lammps_result,
        )
        self._results[job_id] = sim_result
        return job_id

    def poll(self, job_id: str) -> str:
        if job_id not in self._results:
            return "pending"
        return "completed" if self._results[job_id].success else "failed"

    def collect(self, job_id: str) -> SimulationResult:
        return self._results[job_id]
