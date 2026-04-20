"""Simulation execution backends.

Provides an abstract :class:`JobBackend` interface and concrete
implementations for local (subprocess) and Slurm (HPC) execution.
"""

from .base import JobBackend
from .local import LocalBackend

__all__ = [
    "JobBackend",
    "LocalBackend",
]
