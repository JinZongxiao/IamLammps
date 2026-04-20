"""Unified task/result schema for simulation backends.

Provides a backend-agnostic interface for submitting and collecting
simulation results.  Existing code (agent.py, pipeline.py) is **not**
required to use these types — they exist for new code paths that want
to go through the backend abstraction layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SimulationTask:
    """A simulation job to be submitted to a backend."""

    code: str                                           # LAMMPS script text
    backend: str = "local"                              # "local" | "slurm"
    resources: dict[str, Any] = field(default_factory=dict)   # e.g. {"mpi_np": 4, "timeout": 300}
    metadata: dict[str, Any] = field(default_factory=dict)    # arbitrary task metadata
    work_dir: Path | None = None                        # if None, backend creates one


@dataclass
class SimulationResult:
    """Unified result returned by any backend."""

    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)     # stdout_path, output_files, ...
    timing: dict[str, Any] = field(default_factory=dict)      # wall_time, timed_out, ...
    errors: list[str] = field(default_factory=list)
    raw_result: Any = None                                     # backend-specific (e.g. LammpsRunResult)
