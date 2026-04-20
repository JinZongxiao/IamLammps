"""Adapter: run simulations through the configured backend.

Existing code (``agent.py``) continues to call ``run_lammps()`` directly.
New code can use :func:`run_via_backend` to get a :class:`SimulationResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends.base import JobBackend
from .backends.local import LocalBackend
from .config import AgentConfig
from .core_types import SimulationTask, SimulationResult


def get_backend(config: AgentConfig) -> JobBackend:
    """Return the backend instance indicated by *config.backend*."""
    if config.backend == "local":
        return LocalBackend(lammps_command=config.lammps_command)
    elif config.backend == "slurm":
        from .backends.slurm import SlurmBackend
        return SlurmBackend()
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")


def run_via_backend(
    config: AgentConfig,
    code: str,
    work_dir: Path,
    resources: dict[str, Any] | None = None,
) -> SimulationResult:
    """High-level: run a simulation through the configured backend."""
    backend = get_backend(config)
    task = SimulationTask(
        code=code,
        backend=config.backend,
        resources=resources or {
            "mpi_np": config.mpi_np,
            "timeout": config.lammps_timeout,
        },
        work_dir=work_dir,
    )
    return backend.run_sync(task)
