"""Slurm HPC backend — placeholder implementation.

All methods raise :class:`NotImplementedError` until Slurm integration
is actually needed.  The interface mirrors :class:`JobBackend` so that
code can be written against the backend abstraction today.
"""

from __future__ import annotations

from ..core_types import SimulationTask, SimulationResult
from .base import JobBackend


class SlurmBackend(JobBackend):
    """Submit LAMMPS jobs via ``sbatch`` and poll via ``squeue``/``sacct``.

    Parameters
    ----------
    partition : str
        Slurm partition name.
    account : str | None
        Slurm account for job charging.
    qos : str | None
        Quality-of-service level.
    """

    def __init__(
        self,
        partition: str = "default",
        account: str | None = None,
        qos: str | None = None,
    ) -> None:
        self.partition = partition
        self.account = account
        self.qos = qos

    def submit(self, task: SimulationTask) -> str:
        raise NotImplementedError(
            "SlurmBackend.submit() is not yet implemented. "
            "Set BACKEND=local or implement Slurm integration."
        )

    def poll(self, job_id: str) -> str:
        raise NotImplementedError(
            "SlurmBackend.poll() is not yet implemented."
        )

    def collect(self, job_id: str) -> SimulationResult:
        raise NotImplementedError(
            "SlurmBackend.collect() is not yet implemented."
        )
