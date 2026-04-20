"""Uniaxial tensile test workflow for BCC tungsten."""

from __future__ import annotations

import random

from ..pipeline import PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig


class TensileWorkflow(BaseWorkflow):
    """3-stage pipeline: minimize → equilibrate (NPT) → uniaxial deform."""

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        strain_rate: float = 1e8,
        max_strain: float = 0.3,
        supercell: tuple[int, int, int] = (10, 10, 20),
        deform_direction: str = "z",
    ) -> None:
        super().__init__(config, agent)
        self.strain_rate = strain_rate
        self.max_strain = max_strain
        self.supercell = supercell
        self.deform_direction = deform_direction

    def build_pipeline(self) -> SimulationPipeline:
        stages = [
            PipelineStage(
                name="minimize",
                task_description=self._minimize_task(),
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="equilibrate",
                task_description=self._equilibrate_task(),
                depends_on="minimize",
                input_files=["restart.minimize"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="deform",
                task_description=self._deform_task(),
                depends_on="equilibrate",
                input_files=["restart.equilibrate"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    def _minimize_task(self) -> str:
        nx, ny, nz = self.supercell
        return (
            f"Create a BCC tungsten supercell ({nx}x{ny}x{nz} unit cells) "
            f"using metal units and atom_style atomic. "
            f"Lattice constant 3.165 Angstrom. "
            f"Use pair_style eam/alloy with W_zhou.eam.alloy potential file. "
            f"mass 1 183.84. "
            f"Perform energy minimization using CG method with "
            f"etol=1e-10, ftol=1e-10, maxiter=10000, maxeval=100000. "
            f"Output thermo every 100 steps. "
            f"After minimization, write: write_restart restart.minimize"
        )

    def _equilibrate_task(self) -> str:
        seed = random.randint(10000, 99999)
        T = self.config.temperature
        return (
            f"Read the restart file from previous stage: read_restart restart.minimize\n"
            f"Set up pair_style eam/alloy and pair_coeff * * W_zhou.eam.alloy W. "
            f"mass 1 183.84.\n"
            f"Give atoms initial velocities: velocity all create {T} {seed} dist gaussian\n"
            f"Equilibrate using NPT at temperature {T} K and pressure 0 bar for 20000 steps. "
            f"Use fix npt with Tdamp=0.1 and Pdamp=1.0.\n"
            f"timestep 0.001\n"
            f"Output thermo every 100 steps with: "
            f"thermo_style custom step temp press pe ke etotal vol lx ly lz\n"
            f"After equilibration, write: write_restart restart.equilibrate"
        )

    def _deform_task(self) -> str:
        T = self.config.temperature
        d = self.deform_direction
        seed = random.randint(10000, 99999)

        # Calculate LAMMPS erate from physical strain rate.
        # In metal units: erate for fix deform is in 1/ps.
        # strain_rate is in 1/s, so erate = strain_rate * 1e-12.
        erate = self.strain_rate * 1e-12

        return (
            f"Read the restart file: read_restart restart.equilibrate\n"
            f"Set up pair_style eam/alloy and pair_coeff * * W_zhou.eam.alloy W. "
            f"mass 1 183.84.\n"
            f"Apply uniaxial tensile deformation along the {d}-axis.\n"
            f"Use fix deform with erate {erate:.6e} on the {d}-dimension, "
            f"and remap atoms. Allow the other two dimensions to relax "
            f"(use fix npt for temperature {T} K and zero pressure on the "
            f"non-deformed axes).\n"
            f"timestep 0.001\n"
            f"Compute stress: thermo_style custom step temp "
            f"pxx pyy pzz pxy pxz pyz lx ly lz pe ke etotal\n"
            f"Output thermo every 50 steps.\n"
            f"Run for enough steps to reach ~{self.max_strain*100:.0f}% strain. "
            f"With erate={erate:.6e}/ps and timestep=0.001 ps, "
            f"that is approximately {int(self.max_strain / (erate * 0.001))} steps.\n"
            f"Also dump atom positions every 1000 steps for visualization: "
            f"dump 1 all custom 1000 dump.tensile.*.lammpstrj id type x y z"
        )
