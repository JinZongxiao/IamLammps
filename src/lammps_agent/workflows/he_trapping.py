"""Helium trapping and bubble formation workflow."""

from __future__ import annotations

import random

from ..pipeline import PipelineStage, SimulationPipeline
from ..potentials import PotentialManager
from .base import BaseWorkflow, WorkflowConfig


class HeTrappingWorkflow(BaseWorkflow):
    """Multi-stage pipeline for He trapping in W.

    Stages
    ------
    1. **create_defects** — Build W supercell with vacancies.
    2. **minimize** — Relax the defective structure.
    3. **insert_he** — Gradually insert He atoms at defect sites.
    4. **anneal** — MD annealing to allow He migration/clustering.
    5. **analyze** — Dump atomic data for post-processing (coordination, Voronoi).

    Parameters
    ----------
    n_vacancies : int
        Number of vacancies to introduce.
    n_he_atoms : int
        Number of He atoms to insert.
    anneal_temperature : float
        Annealing temperature in K.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        agent: "LammpsAgent",  # noqa: F821
        supercell: tuple[int, int, int] = (10, 10, 10),
        n_vacancies: int = 10,
        n_he_atoms: int = 20,
        anneal_temperature: float = 1000.0,
    ) -> None:
        super().__init__(config, agent)
        self.supercell = supercell
        self.n_vacancies = n_vacancies
        self.n_he_atoms = n_he_atoms
        self.anneal_temperature = anneal_temperature

    def build_pipeline(self) -> SimulationPipeline:
        stages = [
            PipelineStage(
                name="create_defects",
                task_description=self._create_defects_task(),
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="minimize",
                task_description=self._minimize_task(),
                depends_on="create_defects",
                input_files=["restart.defects"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="insert_he",
                task_description=self._insert_he_task(),
                depends_on="minimize",
                input_files=["restart.minimized"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="anneal",
                task_description=self._anneal_task(),
                depends_on="insert_he",
                input_files=["restart.he_inserted"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
            PipelineStage(
                name="analyze",
                task_description=self._analyze_task(),
                depends_on="anneal",
                input_files=["restart.annealed"],
                timeout=self.config.timeout_per_stage,
                potential_name=self.config.potential_name,
            ),
        ]
        return SimulationPipeline(stages, self.agent, PotentialManager())

    # ------------------------------------------------------------------
    # Task descriptions
    # ------------------------------------------------------------------

    def _create_defects_task(self) -> str:
        nx, ny, nz = self.supercell
        return (
            f"Create a BCC tungsten supercell ({nx}x{ny}x{nz} unit cells) "
            f"using metal units, atom_style atomic.\n"
            f"Lattice constant 3.165 Angstrom. mass 1 183.84.\n"
            f"Introduce {self.n_vacancies} vacancies by randomly deleting atoms.\n"
            f"Use pair_style eam/alloy with W_zhou.eam.alloy.\n"
            f"Output thermo every 100 steps.\n"
            f"After construction, write: write_restart restart.defects"
        )

    def _minimize_task(self) -> str:
        return (
            "Read the restart file: read_restart restart.defects\n"
            "Set up pair_style eam/alloy and pair_coeff. mass 1 183.84.\n"
            "Perform energy minimization: etol=1e-10, ftol=1e-10, "
            "maxiter=20000, maxeval=200000.\n"
            "Output thermo every 100 steps.\n"
            "After minimization, write: write_restart restart.minimized"
        )

    def _insert_he_task(self) -> str:
        return (
            "Read the restart file: read_restart restart.minimized\n"
            "Set up the potential for W-He interaction.\n"
            f"Insert {self.n_he_atoms} He atoms at or near vacancy sites.\n"
            "Use create_atoms with appropriate positioning.\n"
            "Set mass for He (type 2): mass 2 4.0026.\n"
            "Perform a short minimization to relax He positions.\n"
            "After insertion, write: write_restart restart.he_inserted"
        )

    def _anneal_task(self) -> str:
        T = self.anneal_temperature
        seed = random.randint(10000, 99999)
        return (
            "Read the restart file: read_restart restart.he_inserted\n"
            "Set up the W-He potential.\n"
            f"velocity all create {T} {seed} dist gaussian\n"
            f"Anneal at {T} K using NVT ensemble for 50000 steps.\n"
            "timestep 0.001\n"
            "Output thermo every 500 steps.\n"
            "After annealing, write: write_restart restart.annealed"
        )

    def _analyze_task(self) -> str:
        return (
            "Read the restart file: read_restart restart.annealed\n"
            "Set up the potential.\n"
            "Compute coordination numbers: compute coord all coord/atom cutoff 3.5\n"
            "Dump atomic data for post-processing:\n"
            "dump 1 all custom 1 dump.he_final.lammpstrj id type x y z c_coord\n"
            "run 0"
        )
