"""Abstract base class for simulation execution backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..core_types import SimulationTask, SimulationResult


class JobBackend(ABC):
    """Backend interface: submit / poll / collect."""

    @abstractmethod
    def submit(self, task: SimulationTask) -> str:
        """Submit a task and return a job-id string."""
        ...

    @abstractmethod
    def poll(self, job_id: str) -> str:
        """Return status: ``'pending'`` | ``'running'`` | ``'completed'`` | ``'failed'``."""
        ...

    @abstractmethod
    def collect(self, job_id: str) -> SimulationResult:
        """Collect results for a completed/failed job."""
        ...

    def run_sync(self, task: SimulationTask) -> SimulationResult:
        """Convenience: submit → poll until done → collect."""
        job_id = self.submit(task)
        while True:
            status = self.poll(job_id)
            if status in ("completed", "failed"):
                break
            time.sleep(0.1)
        return self.collect(job_id)
