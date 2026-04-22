"""Mechanical test workflows package."""

from __future__ import annotations

from .base import BaseWorkflow, WorkflowConfig
from .tensile import TensileWorkflow
from .nanoindent import NanoindentWorkflow
from .crack import CrackWorkflow
from .dbtt_sweep import DBTTSweepWorkflow
from .interface_mechanics import InterfaceMechanicsWorkflow
from .he_trapping import HeTrappingWorkflow

__all__ = [
    "BaseWorkflow",
    "WorkflowConfig",
    "TensileWorkflow",
    "NanoindentWorkflow",
    "CrackWorkflow",
    "DBTTSweepWorkflow",
    "InterfaceMechanicsWorkflow",
    "HeTrappingWorkflow",
    "build_workflow",
]


def build_workflow(
    name: str,
    config: WorkflowConfig,
    agent: "LammpsAgent",  # noqa: F821
    supercell: tuple[int, int, int] = (10, 10, 20),
    strain_rate: float = 1e8,
    # Production parameters (passed through to applicable workflows)
    equilibration_steps: int | None = None,
    timestep: float | None = None,
    thermo_every: int | None = None,
    dump_every: int | None = None,
) -> BaseWorkflow:
    """Factory: create a workflow by name."""
    from ..agent import LammpsAgent  # avoid circular

    if name == "tensile":
        return TensileWorkflow(
            config, agent, supercell=supercell, strain_rate=strain_rate,
            equilibration_steps=equilibration_steps,
            timestep=timestep,
            thermo_every=thermo_every,
            dump_every=dump_every,
        )
    elif name == "nanoindent":
        return NanoindentWorkflow(config, agent, supercell=supercell,
                                  dump_every=dump_every)
    elif name == "crack":
        return CrackWorkflow(config, agent, supercell=supercell,
                             dump_every=dump_every)
    elif name == "dbtt":
        return DBTTSweepWorkflow(
            config, agent, supercell=supercell, strain_rate=strain_rate,
            equilibration_steps=equilibration_steps,
            timestep=timestep,
            thermo_every=thermo_every,
            dump_every=dump_every,
        )
    elif name == "interface":
        return InterfaceMechanicsWorkflow(config, agent, supercell=supercell)
    elif name == "he_trapping":
        return HeTrappingWorkflow(config, agent, supercell=supercell)
    else:
        raise ValueError(
            f"Unknown workflow: {name}. "
            f"Available: tensile, nanoindent, crack, dbtt, interface, he_trapping"
        )
