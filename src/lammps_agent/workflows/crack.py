"""Crack propagation workflow for BCC tungsten."""

from __future__ import annotations

import random

from ..pipeline import PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig


class CrackWorkflow(BaseWorkflow):
    """4-stage: build cracked model → minimize → Mode I loading → analyze."""

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        supercell: tuple[int, int, int] = (30, 15, 5),
        crack_length_fraction: float = 0.3,
    ) -> None:
        super().__init__(config, agent)
        self.supercell = supercell
        self.crack_length_fraction = crack_length_fraction

    def build_pipeline(self) -> SimulationPipeline:
        stages = [
            PipelineStage(
                name="build_crack",
                task_description=self._build_task(),
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="minimize",
                task_description=self._minimize_task(),
                depends_on="build_crack",
                input_files=["restart.cracked"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="load",
                task_description=self._load_task(),
                depends_on="minimize",
                input_files=["restart.minimized"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="analyze",
                task_description=self._analyze_task(),
                depends_on="load",
                input_files=["restart.loaded"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    def _build_task(self) -> str:
        nx, ny, nz = self.supercell
        frac = self.crack_length_fraction
        ly = ny * 3.165
        half_w = 0.01 * ly  # thin notch

        return (
            f"Create BCC tungsten specimen ({nx}x{ny}x{nz}) for crack propagation.\n"
            f"Metal units, atom_style atomic, lattice bcc 3.165, boundary p p p.\n"
            f"pair_style eam/alloy, pair_coeff * * W_zhou.eam.alloy W, mass 1 183.84.\n"
            f"Create a pre-crack by deleting atoms in a thin notch region:\n"
            f"  region crack block 0 {frac*nx*3.165:.2f} "
            f"{ly/2 - half_w:.2f} {ly/2 + half_w:.2f} INF INF\n"
            f"  delete_atoms region crack\n"
            f"Define boundary groups:\n"
            f"  'upper' group: top 10% of y\n"
            f"  'lower' group: bottom 10% of y\n"
            f"  'mobile' group: everything else\n"
            f"write_restart restart.cracked"
        )

    def _minimize_task(self) -> str:
        return (
            f"read_restart restart.cracked\n"
            f"pair_style eam/alloy, pair_coeff * * W_zhou.eam.alloy W, mass 1 183.84.\n"
            f"Minimize the cracked specimen: CG, etol=1e-8, ftol=1e-8, maxiter=20000, maxeval=200000.\n"
            f"thermo 100\n"
            f"write_restart restart.minimized"
        )

    def _load_task(self) -> str:
        T = self.config.temperature
        seed = random.randint(10000, 99999)
        # Mode I loading: pull upper and lower boundaries apart in y-direction.
        return (
            f"read_restart restart.minimized\n"
            f"pair_style eam/alloy, pair_coeff * * W_zhou.eam.alloy W, mass 1 183.84.\n"
            f"Re-create groups: upper (top 10% y), lower (bottom 10% y), mobile (rest).\n"
            f"velocity mobile create {T} {seed} dist gaussian\n"
            f"Mode I loading: apply displacement boundary conditions.\n"
            f"  fix pull_up upper_atoms aveforce 0 0.001 0\n"
            f"  fix pull_down lower_atoms aveforce 0 -0.001 0\n"
            f"Or use fix deform with erate on y-dimension.\n"
            f"NVE for mobile atoms. timestep 0.001.\n"
            f"thermo_style custom step temp pyy ly pe\n"
            f"thermo 100\n"
            f"Run 50000 steps.\n"
            f"dump 1 all custom 1000 dump.crack.*.lammpstrj id type x y z\n"
            f"compute cna all cna/atom 3.5\n"
            f"write_restart restart.loaded"
        )

    def _analyze_task(self) -> str:
        return (
            f"read_restart restart.loaded\n"
            f"pair_style eam/alloy, pair_coeff * * W_zhou.eam.alloy W, mass 1 183.84.\n"
            f"Compute coordination number and CNA to identify crack tip position.\n"
            f"compute coord all coord/atom cutoff 3.5\n"
            f"compute cna all cna/atom 3.5\n"
            f"dump 1 all custom 1 dump.crack_final.lammpstrj id type x y z c_coord c_cna\n"
            f"run 0"
        )
