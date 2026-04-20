"""Interface mechanical testing workflow for oxide/W interfaces."""

from __future__ import annotations

import random

from ..pipeline import PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig


class InterfaceMechanicsWorkflow(BaseWorkflow):
    """Multi-stage pipeline for oxide/W interface mechanical properties.

    Stages
    ------
    1. **build_interface** — Construct oxide/W bilayer slab.
    2. **minimize** — Energy minimization of the interface structure.
    3. **equilibrate** — NPT equilibration at target temperature.
    4. **tensile_normal** — Pull perpendicular to interface (adhesion test).

    Parameters
    ----------
    oxide_type : str
        Oxide phase identifier (e.g. ``"Y2O3"``, ``"Al2O3"``).
    interface_plane : str
        W surface orientation, e.g. ``"(110)"``.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        supercell: tuple[int, int, int] = (10, 10, 20),
        oxide_type: str = "Y2O3",
        interface_plane: str = "(110)",
    ) -> None:
        super().__init__(config, agent)
        self.supercell = supercell
        self.oxide_type = oxide_type
        self.interface_plane = interface_plane

    def build_pipeline(self) -> SimulationPipeline:
        stages = [
            PipelineStage(
                name="build_interface",
                task_description=self._build_task(),
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="minimize",
                task_description=self._minimize_task(),
                depends_on="build_interface",
                input_files=["restart.interface", "data.interface"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="equilibrate",
                task_description=self._equilibrate_task(),
                depends_on="minimize",
                input_files=["restart.minimized"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="tensile_normal",
                task_description=self._tensile_task(),
                depends_on="equilibrate",
                input_files=["restart.equilibrate"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    # ------------------------------------------------------------------
    # Task descriptions for each stage
    # ------------------------------------------------------------------

    def _build_task(self) -> str:
        nx, ny, nz = self.supercell
        return (
            f"Build an oxide/W interface slab model.\n"
            f"Use a BCC W slab with {self.interface_plane} surface orientation, "
            f"supercell size {nx}x{ny}x{nz//2} unit cells.\n"
            f"Place a {self.oxide_type} oxide layer on top of the W slab.\n"
            f"Use metal units, atom_style atomic, boundary p p s.\n"
            f"After construction, write: write_restart restart.interface\n"
            f"Also write: write_data data.interface"
        )

    def _minimize_task(self) -> str:
        return (
            "Read the restart file: read_restart restart.interface\n"
            "Set up the appropriate pair_style and pair_coeff.\n"
            "Perform energy minimization with etol=1e-8, ftol=1e-8, "
            "maxiter=20000, maxeval=200000.\n"
            "Output thermo every 100 steps.\n"
            "After minimization, write: write_restart restart.minimized"
        )

    def _equilibrate_task(self) -> str:
        T = self.config.temperature
        seed = random.randint(10000, 99999)
        return (
            f"Read the restart file: read_restart restart.minimized\n"
            f"Set up the pair_style and pair_coeff.\n"
            f"velocity all create {T} {seed} dist gaussian\n"
            f"Equilibrate using NPT at temperature {T} K and pressure 0 bar "
            f"for 20000 steps.\n"
            f"timestep 0.001\n"
            f"Output thermo every 100 steps.\n"
            f"After equilibration, write: write_restart restart.equilibrate"
        )

    def _tensile_task(self) -> str:
        T = self.config.temperature
        return (
            f"Read the restart file: read_restart restart.equilibrate\n"
            f"Set up the pair_style and pair_coeff.\n"
            f"Apply uniaxial tensile deformation perpendicular to the interface "
            f"(z-direction) to measure interface adhesion.\n"
            f"Use fix deform with a small erate on the z-dimension.\n"
            f"Keep temperature at {T} K.\n"
            f"Output thermo every 50 steps including: "
            f"step temp pxx pyy pzz lx ly lz pe ke etotal\n"
            f"Run for enough steps to observe interface separation."
        )
